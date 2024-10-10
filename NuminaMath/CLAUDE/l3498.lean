import Mathlib

namespace super_ball_distance_l3498_349889

def initial_height : ℚ := 80
def rebound_factor : ℚ := 2/3
def num_bounces : ℕ := 4

def bounce_sequence (n : ℕ) : ℚ :=
  initial_height * (rebound_factor ^ n)

def total_distance : ℚ :=
  2 * (initial_height * (1 - rebound_factor^(num_bounces + 1)) / (1 - rebound_factor)) - initial_height

theorem super_ball_distance :
  total_distance = 11280/81 :=
sorry

end super_ball_distance_l3498_349889


namespace set_operations_l3498_349849

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Define the theorem
theorem set_operations :
  (A ∩ B = Ioc (-3) 2) ∧
  (A ∪ B = Ioc (-5) 3) ∧
  (Aᶜ = Iic (-5) ∪ Ici 2) ∧
  ((A ∩ B)ᶜ = Iic (-3) ∪ Ioi 2) ∧
  (Aᶜ ∩ B = Icc 2 3) :=
by sorry

end set_operations_l3498_349849


namespace expression_simplification_l3498_349863

theorem expression_simplification (m : ℝ) 
  (h1 : (m + 2) * (m - 3) = 0) 
  (h2 : m ≠ 3) : 
  ((m^2 - 9) / (m^2 - 6*m + 9) - 3 / (m - 3)) / (m^2 / m^3) = -4/5 := by
  sorry

end expression_simplification_l3498_349863


namespace dress_shop_inventory_l3498_349878

theorem dress_shop_inventory (total_space : ℕ) (blue_extra : ℕ) (red_dresses : ℕ) 
  (h1 : total_space = 200)
  (h2 : blue_extra = 34)
  (h3 : red_dresses + (red_dresses + blue_extra) = total_space) :
  red_dresses = 83 := by
sorry

end dress_shop_inventory_l3498_349878


namespace keith_cantaloupes_l3498_349899

/-- The number of cantaloupes grown by Keith, given the total number of cantaloupes
    and the numbers grown by Fred and Jason. -/
theorem keith_cantaloupes (total : ℕ) (fred : ℕ) (jason : ℕ) 
    (h_total : total = 65) 
    (h_fred : fred = 16) 
    (h_jason : jason = 20) : 
  total - (fred + jason) = 29 := by
  sorry

end keith_cantaloupes_l3498_349899


namespace set_difference_equals_singleton_l3498_349866

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}

def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

theorem set_difference_equals_singleton : N \ M = {2003} := by
  sorry

end set_difference_equals_singleton_l3498_349866


namespace parametric_to_general_plane_equation_l3498_349875

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  origin : Point3D
  dir1 : Point3D
  dir2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a plane equation satisfies the required conditions -/
def isValidPlaneEquation (eq : PlaneEquation) : Prop :=
  eq.A > 0 ∧ Int.gcd (Int.natAbs eq.A) (Int.gcd (Int.natAbs eq.B) (Int.gcd (Int.natAbs eq.C) (Int.natAbs eq.D))) = 1

/-- The main theorem stating the equivalence of the parametric and general forms of the plane -/
theorem parametric_to_general_plane_equation 
  (plane : ParametricPlane)
  (h_plane : plane = { 
    origin := { x := 3, y := 4, z := 1 },
    dir1 := { x := 1, y := -2, z := -1 },
    dir2 := { x := 2, y := 0, z := 1 }
  }) :
  ∃ (eq : PlaneEquation), 
    isValidPlaneEquation eq ∧
    (∀ (p : Point3D), 
      (∃ (s t : ℝ), 
        p.x = plane.origin.x + s * plane.dir1.x + t * plane.dir2.x ∧
        p.y = plane.origin.y + s * plane.dir1.y + t * plane.dir2.y ∧
        p.z = plane.origin.z + s * plane.dir1.z + t * plane.dir2.z) ↔
      eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0) ∧
    eq = { A := 2, B := 3, C := -4, D := -14 } :=
  sorry

end parametric_to_general_plane_equation_l3498_349875


namespace angle_c_measure_l3498_349869

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the properties of the isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.A = t.B

-- Define the relationship between angles A and C
def AngleCRelation (t : Triangle) : Prop :=
  t.C = t.A + 30

-- Define the sum of angles in a triangle
def AngleSum (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180

-- Theorem statement
theorem angle_c_measure (t : Triangle) 
  (h1 : IsIsosceles t) 
  (h2 : AngleCRelation t) 
  (h3 : AngleSum t) : 
  t.C = 80 := by
    sorry

end angle_c_measure_l3498_349869


namespace child_ticket_cost_l3498_349818

/-- Proves that the cost of a child ticket is $5 given the theater conditions --/
theorem child_ticket_cost (total_seats : ℕ) (adult_price : ℕ) (child_tickets : ℕ) (total_revenue : ℕ) :
  total_seats = 80 →
  adult_price = 12 →
  child_tickets = 63 →
  total_revenue = 519 →
  ∃ (child_price : ℕ), 
    child_price = 5 ∧
    total_revenue = (total_seats - child_tickets) * adult_price + child_tickets * child_price :=
by
  sorry

end child_ticket_cost_l3498_349818


namespace cricket_ratio_l3498_349816

/-- Represents the number of crickets Spike hunts in the morning -/
def morning_crickets : ℕ := 5

/-- Represents the total number of crickets Spike hunts per day -/
def total_crickets : ℕ := 20

/-- Represents the number of crickets Spike hunts in the afternoon and evening -/
def afternoon_evening_crickets : ℕ := total_crickets - morning_crickets

/-- The theorem stating the ratio of crickets hunted in the afternoon and evening to morning -/
theorem cricket_ratio : 
  afternoon_evening_crickets / morning_crickets = 3 :=
sorry

end cricket_ratio_l3498_349816


namespace min_value_expression_l3498_349862

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 := by
sorry

end min_value_expression_l3498_349862


namespace lion_king_cost_l3498_349888

theorem lion_king_cost (
  lion_king_earnings : ℝ)
  (star_wars_cost : ℝ)
  (star_wars_earnings : ℝ)
  (h1 : lion_king_earnings = 200)
  (h2 : star_wars_cost = 25)
  (h3 : star_wars_earnings = 405)
  (h4 : lion_king_earnings - (lion_king_earnings - (star_wars_earnings - star_wars_cost) / 2) = 10) :
  ∃ (lion_king_cost : ℝ), lion_king_cost = 10 := by
  sorry

end lion_king_cost_l3498_349888


namespace soda_can_ratio_l3498_349829

theorem soda_can_ratio :
  let initial_cans : ℕ := 22
  let taken_cans : ℕ := 6
  let final_cans : ℕ := 24
  let remaining_cans := initial_cans - taken_cans
  let bought_cans := final_cans - remaining_cans
  (bought_cans : ℚ) / remaining_cans = 1 / 2 :=
by
  sorry

end soda_can_ratio_l3498_349829


namespace quadratic_function_properties_l3498_349847

/-- A quadratic function f(x) = ax^2 + (b-2)x + 3 where a ≠ 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem quadratic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  /- If the solution set of f(x) > 0 is (-1, 3), then a = -1 and b = 4 -/
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (a = -1 ∧ b = 4) ∧
  /- If f(1) = 2, a > 0, and b > 0, then the minimum value of 1/a + 4/b is 9 -/
  (f a b 1 = 2 ∧ a > 0 ∧ b > 0 →
   ∀ a' b', a' > 0 → b' > 0 → f a' b' 1 = 2 → 1/a' + 4/b' ≥ 9) :=
by sorry

end quadratic_function_properties_l3498_349847


namespace sin_four_thirds_pi_l3498_349886

theorem sin_four_thirds_pi : Real.sin (4 / 3 * Real.pi) = -(Real.sqrt 3 / 2) := by
  sorry

end sin_four_thirds_pi_l3498_349886


namespace line_intersecting_circle_slope_l3498_349834

/-- A line passing through (4,0) intersecting the circle (x-2)^2 + y^2 = 1 has slope -√3/3 or √3/3 -/
theorem line_intersecting_circle_slope :
  ∀ (k : ℝ), 
    (∃ (x y : ℝ), y = k * (x - 4) ∧ (x - 2)^2 + y^2 = 1) →
    (k = -Real.sqrt 3 / 3 ∨ k = Real.sqrt 3 / 3) :=
by sorry

end line_intersecting_circle_slope_l3498_349834


namespace intersection_M_N_l3498_349807

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_M_N_l3498_349807


namespace special_circle_equation_l3498_349895

/-- A circle with center on y = 2x and specific chord lengths -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 2 * center.1
  x_chord_length : 4 = 2 * (radius ^ 2 - center.1 ^ 2).sqrt
  y_chord_length : 8 = 2 * (radius ^ 2 - center.2 ^ 2).sqrt

/-- The equation of the circle is one of two specific forms -/
theorem special_circle_equation (c : SpecialCircle) :
  (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 5 ↔ (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) ∨
  (∀ x y : ℝ, (x + 1) ^ 2 + (y + 2) ^ 2 = 5 ↔ (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2) :=
sorry

end special_circle_equation_l3498_349895


namespace total_sandwiches_l3498_349826

/-- Represents the number of sandwiches of each type -/
structure Sandwiches where
  cheese : ℕ
  bologna : ℕ
  peanutButter : ℕ

/-- The ratio of sandwich types -/
def sandwichRatio : Sandwiches :=
  { cheese := 1
    bologna := 7
    peanutButter := 8 }

/-- Theorem: Given the sandwich ratio and the number of bologna sandwiches,
    prove the total number of sandwiches -/
theorem total_sandwiches
    (ratio : Sandwiches)
    (bologna_count : ℕ)
    (h1 : ratio = sandwichRatio)
    (h2 : bologna_count = 35) :
    ratio.cheese * (bologna_count / ratio.bologna) +
    bologna_count +
    ratio.peanutButter * (bologna_count / ratio.bologna) = 80 := by
  sorry

#check total_sandwiches

end total_sandwiches_l3498_349826


namespace soda_bottle_difference_l3498_349839

theorem soda_bottle_difference (regular_soda : ℕ) (diet_soda : ℕ)
  (h1 : regular_soda = 81)
  (h2 : diet_soda = 60) :
  regular_soda - diet_soda = 21 := by
  sorry

end soda_bottle_difference_l3498_349839


namespace intersection_M_N_l3498_349821

def M : Set ℝ := {0, 1, 2}

def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end intersection_M_N_l3498_349821


namespace system_solution_l3498_349855

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (eq1 : 2*x₁ + x₂ + x₃ + x₄ + x₅ = 6)
  (eq2 : x₁ + 2*x₂ + x₃ + x₄ + x₅ = 12)
  (eq3 : x₁ + x₂ + 2*x₃ + x₄ + x₅ = 24)
  (eq4 : x₁ + x₂ + x₃ + 2*x₄ + x₅ = 48)
  (eq5 : x₁ + x₂ + x₃ + x₄ + 2*x₅ = 96) :
  3*x₄ + 2*x₅ = 181 := by
sorry

end system_solution_l3498_349855


namespace quadratic_equation_factorization_l3498_349813

theorem quadratic_equation_factorization (n : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 - 4*x + 1 = n ↔ (x - m)^2 = 5) →
  n = 2 :=
by sorry

end quadratic_equation_factorization_l3498_349813


namespace min_S_19_l3498_349809

/-- Given an arithmetic sequence {a_n} where S_8 ≤ 6 and S_11 ≥ 27, 
    the minimum value of S_19 is 133. -/
theorem min_S_19 (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 8 ≤ 6 →                                 -- Given condition
  S 11 ≥ 27 →                               -- Given condition
  ∀ S_19 : ℝ, (S_19 = S 19 → S_19 ≥ 133) :=
by sorry

end min_S_19_l3498_349809


namespace sequence_sum_l3498_349835

theorem sequence_sum (n : ℕ) (x : ℕ → ℚ) : 
  (∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + 1 / 3) →
  x 1 = 2 →
  n > 0 →
  Finset.sum (Finset.range n) (λ i => x (i + 1)) = n * (n + 11) / 6 :=
by sorry

end sequence_sum_l3498_349835


namespace yunhwan_water_consumption_l3498_349838

/-- Yunhwan's yearly water consumption in liters -/
def yearly_water_consumption (monthly_consumption : ℝ) (months_per_year : ℕ) : ℝ :=
  monthly_consumption * months_per_year

/-- Proof that Yunhwan's yearly water consumption is 2194.56 liters -/
theorem yunhwan_water_consumption : 
  yearly_water_consumption 182.88 12 = 2194.56 := by
  sorry

end yunhwan_water_consumption_l3498_349838


namespace fraction_product_simplification_l3498_349882

theorem fraction_product_simplification :
  (150 : ℚ) / 12 * 7 / 140 * 6 / 5 = 3 / 4 := by
  sorry

end fraction_product_simplification_l3498_349882


namespace range_of_a_l3498_349824

/-- Proposition p: The equation x^2 + ax + 1 = 0 has solutions -/
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x + 1 = 0

/-- Proposition q: For all x ∈ ℝ, e^(2x) - 2e^x + a ≥ 0 always holds -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, Real.exp (2*x) - 2*(Real.exp x) + a ≥ 0

/-- The range of a given p ∧ q is true -/
theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ∈ Set.Ici (0 : ℝ) := by
  sorry

#check range_of_a

end range_of_a_l3498_349824


namespace taehyung_candies_l3498_349842

def total_candies : ℕ := 6
def seokjin_eats : ℕ := 4

theorem taehyung_candies : total_candies - seokjin_eats = 2 := by
  sorry

end taehyung_candies_l3498_349842


namespace base_conversion_536_7_to_6_l3498_349885

/-- Converts a number from base b1 to base b2 -/
def convert_base (n : ℕ) (b1 b2 : ℕ) : ℕ :=
  sorry

/-- Checks if a number n in base b has the given digits -/
def check_digits (n : ℕ) (b : ℕ) (digits : List ℕ) : Prop :=
  sorry

theorem base_conversion_536_7_to_6 :
  convert_base 536 7 6 = 1132 ∧ 
  check_digits 536 7 [6, 3, 5] ∧
  check_digits 1132 6 [2, 3, 1, 1] :=
sorry

end base_conversion_536_7_to_6_l3498_349885


namespace water_bill_calculation_l3498_349860

/-- Water bill calculation for a household --/
theorem water_bill_calculation 
  (a : ℝ) -- Base rate for water usage up to 20 cubic meters
  (usage : ℝ) -- Total water usage
  (h1 : usage = 25) -- The household used 25 cubic meters
  (h2 : usage > 20) -- Usage exceeds 20 cubic meters
  : 
  (min usage 20) * a + (usage - 20) * (a + 3) = 25 * a + 15 :=
by sorry

end water_bill_calculation_l3498_349860


namespace complex_modulus_l3498_349867

theorem complex_modulus (z : ℂ) : i * z = Real.sqrt 2 - i → Complex.abs z = Real.sqrt 3 := by
  sorry

end complex_modulus_l3498_349867


namespace square_diagonal_characterization_l3498_349891

/-- A quadrilateral with vertices A, B, C, and D in 2D space. -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The diagonals of a quadrilateral. -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.C.1 - q.A.1, q.C.2 - q.A.2), (q.D.1 - q.B.1, q.D.2 - q.B.2))

/-- Check if two vectors are perpendicular. -/
def are_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Check if a vector bisects another vector. -/
def bisects (v w : ℝ × ℝ) : Prop :=
  v.1 = w.1 / 2 ∧ v.2 = w.2 / 2

/-- Check if two vectors have equal length. -/
def equal_length (v w : ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 = w.1^2 + w.2^2

/-- A square is a quadrilateral with all sides equal and all angles right angles. -/
def is_square (q : Quadrilateral) : Prop :=
  let (AC, BD) := diagonals q
  are_perpendicular AC BD ∧
  bisects AC BD ∧
  bisects BD AC ∧
  equal_length AC BD

theorem square_diagonal_characterization (q : Quadrilateral) :
  is_square q ↔
    let (AC, BD) := diagonals q
    are_perpendicular AC BD ∧
    bisects AC BD ∧
    bisects BD AC ∧
    equal_length AC BD :=
  sorry

end square_diagonal_characterization_l3498_349891


namespace cycling_distance_l3498_349864

theorem cycling_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 8 → time = 2.25 → distance = rate * time → distance = 18 := by
sorry

end cycling_distance_l3498_349864


namespace prob_two_tails_two_heads_proof_l3498_349856

/-- The probability of getting exactly two tails and two heads when four fair coins are tossed simultaneously -/
def prob_two_tails_two_heads : ℚ := 3/8

/-- The number of ways to choose 2 items from 4 items -/
def choose_two_from_four : ℕ := 6

/-- The probability of a specific sequence of two tails and two heads -/
def prob_specific_sequence : ℚ := 1/16

theorem prob_two_tails_two_heads_proof :
  prob_two_tails_two_heads = choose_two_from_four * prob_specific_sequence :=
by sorry

end prob_two_tails_two_heads_proof_l3498_349856


namespace shifted_proportional_function_l3498_349803

/-- Given a proportional function y = -2x that is shifted up by 3 units,
    the resulting function is y = -2x + 3. -/
theorem shifted_proportional_function :
  let f : ℝ → ℝ := λ x ↦ -2 * x
  let shift : ℝ := 3
  let g : ℝ → ℝ := λ x ↦ f x + shift
  ∀ x : ℝ, g x = -2 * x + 3 := by
  sorry

end shifted_proportional_function_l3498_349803


namespace correlation_theorem_l3498_349833

-- Define the types for our quantities
def Time := ℝ
def Displacement := ℝ
def Grade := ℝ
def Weight := ℝ
def DrunkDrivers := ℕ
def TrafficAccidents := ℕ
def Volume := ℝ

-- Define a type for pairs of quantities
structure QuantityPair where
  first : Type
  second : Type

-- Define our pairs
def uniformMotionPair : QuantityPair := ⟨Time, Displacement⟩
def gradeWeightPair : QuantityPair := ⟨Grade, Weight⟩
def drunkDriverAccidentPair : QuantityPair := ⟨DrunkDrivers, TrafficAccidents⟩
def volumeWeightPair : QuantityPair := ⟨Volume, Weight⟩

-- Define a predicate for correlation
def hasCorrelation (pair : QuantityPair) : Prop := sorry

-- Theorem statement
theorem correlation_theorem :
  ¬ hasCorrelation uniformMotionPair ∧
  ¬ hasCorrelation gradeWeightPair ∧
  hasCorrelation drunkDriverAccidentPair ∧
  ¬ hasCorrelation volumeWeightPair :=
sorry

end correlation_theorem_l3498_349833


namespace parabola_count_equals_intersection_count_l3498_349894

-- Define the basic geometric objects
structure Line :=
  (a b c : ℝ)

structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Parabola :=
  (focus : Point)
  (directrix : Line)

-- Define the given lines
def t₁ : Line := sorry
def t₂ : Line := sorry
def t₃ : Line := sorry
def e : Line := sorry

-- Define the circumcircle of the triangle formed by t₁, t₂, t₃
def circumcircle : Circle := sorry

-- Function to count intersection points between a circle and a line
def intersectionCount (c : Circle) (l : Line) : Nat := sorry

-- Function to count parabolas touching t₁, t₂, t₃ with focus on e
def parabolaCount : Nat := sorry

-- Theorem statement
theorem parabola_count_equals_intersection_count :
  parabolaCount = intersectionCount circumcircle e :=
sorry

end parabola_count_equals_intersection_count_l3498_349894


namespace solve_for_a_l3498_349876

theorem solve_for_a (A : Set ℝ) (a : ℝ) 
  (h1 : A = {a - 2, 2 * a^2 + 5 * a, 12})
  (h2 : -3 ∈ A) : 
  a = -3/2 := by sorry

end solve_for_a_l3498_349876


namespace bryans_score_l3498_349893

/-- Represents the math exam scores for Bryan, Jen, and Sammy -/
structure ExamScores where
  bryan : ℕ
  jen : ℕ
  sammy : ℕ

/-- The total points possible on the exam -/
def totalPoints : ℕ := 35

/-- Defines the relationship between the scores based on the given conditions -/
def validScores (scores : ExamScores) : Prop :=
  scores.jen = scores.bryan + 10 ∧
  scores.sammy = scores.jen - 2 ∧
  scores.sammy = totalPoints - 7

/-- Theorem stating Bryan's score on the exam -/
theorem bryans_score (scores : ExamScores) (h : validScores scores) : scores.bryan = 20 := by
  sorry

end bryans_score_l3498_349893


namespace inequality_proof_l3498_349854

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c) / (a^2 + b * c) + (c * a) / (b^2 + c * a) + (a * b) / (c^2 + a * b) ≤
  a / (b + c) + b / (c + a) + c / (a + b) := by
  sorry

end inequality_proof_l3498_349854


namespace proportionality_check_l3498_349892

-- Define the concept of direct and inverse proportionality
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k / x

-- Define the equations
def eq_A (x y : ℝ) : Prop := 2*x + 3*y = 5
def eq_B (x y : ℝ) : Prop := 7*x*y = 14
def eq_C (x y : ℝ) : Prop := x = 7*y + 1
def eq_D (x y : ℝ) : Prop := 4*x + 2*y = 8
def eq_E (x y : ℝ) : Prop := x/y = 5

-- Theorem statement
theorem proportionality_check :
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_A x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_D x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_B x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_C x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_E x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) :=
by sorry

end proportionality_check_l3498_349892


namespace inequality_proof_l3498_349808

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_condition : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end inequality_proof_l3498_349808


namespace ice_cream_sundaes_l3498_349848

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end ice_cream_sundaes_l3498_349848


namespace product_plus_one_l3498_349815

theorem product_plus_one (m n : ℕ) (h : m * n = 121) : (m + 1) * (n + 1) = 144 := by
  sorry

end product_plus_one_l3498_349815


namespace absolute_value_inequality_l3498_349823

theorem absolute_value_inequality (a b : ℝ) : 
  (1 / |a| < 1 / |b|) → |a| > |b| := by
  sorry

end absolute_value_inequality_l3498_349823


namespace spiral_strip_length_l3498_349804

/-- The length of a spiral strip on a right circular cylinder -/
theorem spiral_strip_length (base_circumference height : ℝ) 
  (h_base : base_circumference = 18)
  (h_height : height = 8) :
  Real.sqrt (height^2 + base_circumference^2) = Real.sqrt 388 := by
  sorry

end spiral_strip_length_l3498_349804


namespace f_2018_value_l3498_349871

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + x^2017

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| (n+1) => deriv (f_n n)

theorem f_2018_value (x : ℝ) :
  f_n 2018 x = -Real.sin x + Real.exp x := by sorry

end f_2018_value_l3498_349871


namespace family_income_and_tax_calculation_l3498_349897

/-- Family income and tax calculation -/
theorem family_income_and_tax_calculation 
  (father_monthly_income mother_monthly_income grandmother_monthly_pension mikhail_monthly_scholarship : ℕ)
  (property_cadastral_value property_area : ℕ)
  (lada_priora_hp lada_priora_months lada_xray_hp lada_xray_months : ℕ)
  (land_cadastral_value land_area : ℕ)
  (tour_cost_per_person : ℕ)
  (h1 : father_monthly_income = 50000)
  (h2 : mother_monthly_income = 28000)
  (h3 : grandmother_monthly_pension = 15000)
  (h4 : mikhail_monthly_scholarship = 3000)
  (h5 : property_cadastral_value = 6240000)
  (h6 : property_area = 78)
  (h7 : lada_priora_hp = 106)
  (h8 : lada_priora_months = 3)
  (h9 : lada_xray_hp = 122)
  (h10 : lada_xray_months = 8)
  (h11 : land_cadastral_value = 420300)
  (h12 : land_area = 10)
  (h13 : tour_cost_per_person = 17900) :
  ∃ (january_income annual_income property_tax transport_tax land_tax remaining_funds : ℕ),
    january_income = 86588 ∧
    annual_income = 137236 ∧
    property_tax = 4640 ∧
    transport_tax = 3775 ∧
    land_tax = 504 ∧
    remaining_funds = 38817 :=
by sorry


end family_income_and_tax_calculation_l3498_349897


namespace quadratic_roots_sum_l3498_349879

/-- Given a quadratic function f(x) = x^2 + ax + b with roots -2 and 3, prove that a + b = -7 -/
theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = -2 ∨ x = 3) → a + b = -7 := by
  sorry

end quadratic_roots_sum_l3498_349879


namespace x_value_when_one_in_set_l3498_349851

theorem x_value_when_one_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x = -1 := by
  sorry

end x_value_when_one_in_set_l3498_349851


namespace first_half_speed_l3498_349853

/-- Given a trip with the following properties:
  - Total distance is 50 km
  - First half (25 km) is traveled at speed v km/h
  - Second half (25 km) is traveled at 30 km/h
  - Average speed of the entire trip is 40 km/h
  Then the speed v of the first half of the trip is 100/3 km/h. -/
theorem first_half_speed (v : ℝ) : 
  v > 0 → -- Ensure v is positive
  (25 / v + 25 / 30) * 40 = 50 → -- Average speed equation
  v = 100 / 3 := by
sorry


end first_half_speed_l3498_349853


namespace sum_of_roots_quadratic_l3498_349852

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x^2 - 6*x + 8 = (x - r₁) * (x - r₂)) :=
by sorry

end sum_of_roots_quadratic_l3498_349852


namespace probability_face_then_number_standard_deck_l3498_349830

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (face_cards_per_suit : ℕ)
  (number_cards_per_suit : ℕ)

/-- The probability of drawing a face card first and a number card second from a standard deck -/
def probability_face_then_number (d : Deck) : ℚ :=
  let total_face_cards := d.face_cards_per_suit * d.suits
  let total_number_cards := d.number_cards_per_suit * d.suits
  (total_face_cards * total_number_cards : ℚ) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing a face card first and a number card second from a standard deck -/
theorem probability_face_then_number_standard_deck :
  let d : Deck := {
    total_cards := 52,
    ranks := 13,
    suits := 4,
    face_cards_per_suit := 3,
    number_cards_per_suit := 9
  }
  probability_face_then_number d = 8 / 49 := by sorry

end probability_face_then_number_standard_deck_l3498_349830


namespace cylinder_volume_change_l3498_349873

/-- Given a cylinder with volume 15 cubic meters, if its radius is tripled
    and its height is doubled, then its new volume is 270 cubic meters. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 15 → π * (3*r)^2 * (2*h) = 270 := by sorry

end cylinder_volume_change_l3498_349873


namespace max_volume_container_l3498_349841

/-- Represents the dimensions and volume of a rectangular container --/
structure Container where
  shorter_side : ℝ
  longer_side : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the volume of a container given its dimensions --/
def calculate_volume (c : Container) : ℝ :=
  c.shorter_side * c.longer_side * c.height

/-- Defines the constraints for the container based on the problem --/
def is_valid_container (c : Container) : Prop :=
  c.longer_side = c.shorter_side + 0.5 ∧
  c.height = 3.2 - 2 * c.shorter_side ∧
  4 * (c.shorter_side + c.longer_side + c.height) = 14.8 ∧
  c.volume = calculate_volume c

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (c : Container), is_valid_container c ∧
    c.volume = 1.8 ∧
    c.height = 1.2 ∧
    ∀ (c' : Container), is_valid_container c' → c'.volume ≤ c.volume :=
  sorry

end max_volume_container_l3498_349841


namespace cube_root_equation_solution_l3498_349814

theorem cube_root_equation_solution : 
  ∃! x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 :=
by
  -- The unique solution is x = 33
  use 33
  sorry

end cube_root_equation_solution_l3498_349814


namespace simplify_exponents_l3498_349843

theorem simplify_exponents (t s : ℝ) : (t^2 * t^5) * s^3 = t^7 * s^3 := by
  sorry

end simplify_exponents_l3498_349843


namespace bottle_caps_per_box_l3498_349837

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℚ) (caps_per_box : ℕ) :
  total_caps = 245 →
  num_boxes = 7 →
  caps_per_box * num_boxes = total_caps →
  caps_per_box = 35 := by
  sorry

end bottle_caps_per_box_l3498_349837


namespace total_students_l3498_349857

/-- Proves that in a college with a boy-to-girl ratio of 8:5 and 400 girls, the total number of students is 1040 -/
theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 400) : boys + girls = 1040 := by
  sorry

end total_students_l3498_349857


namespace largest_number_in_set_l3498_349802

/-- Given a = -3, -4a is the largest number in the set {-4a, 3a, 36/a, a^3, 2} -/
theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  (-4 * a) = max (-4 * a) (max (3 * a) (max (36 / a) (max (a ^ 3) 2))) :=
by sorry

end largest_number_in_set_l3498_349802


namespace power_relation_l3498_349881

theorem power_relation (a x y : ℝ) (ha : a > 0) (hx : a^x = 2) (hy : a^y = 3) :
  a^(x - 2*y) = 2/9 := by
  sorry

end power_relation_l3498_349881


namespace oliver_seashell_difference_l3498_349896

/-- The number of seashells Oliver collected on Monday -/
def monday_shells : ℕ := 2

/-- The total number of seashells Oliver collected -/
def total_shells : ℕ := 4

/-- The number of seashells Oliver collected on Tuesday -/
def tuesday_shells : ℕ := total_shells - monday_shells

/-- Theorem: Oliver collected 2 more seashells on Tuesday compared to Monday -/
theorem oliver_seashell_difference : tuesday_shells - monday_shells = 2 := by
  sorry

end oliver_seashell_difference_l3498_349896


namespace geometric_series_calculation_l3498_349827

theorem geometric_series_calculation : 
  2016 * (1 / (1 + 1/2 + 1/4 + 1/8 + 1/16 + 1/32)) = 1024 := by
  sorry

end geometric_series_calculation_l3498_349827


namespace cylinder_lateral_area_not_base_area_times_height_l3498_349820

/-- The lateral area of a cylinder is not equal to the base area multiplied by the height. -/
theorem cylinder_lateral_area_not_base_area_times_height 
  (r h : ℝ) (r_pos : 0 < r) (h_pos : 0 < h) :
  2 * π * r * h ≠ (π * r^2) * h := by sorry

end cylinder_lateral_area_not_base_area_times_height_l3498_349820


namespace prism_volume_l3498_349846

/-- A right rectangular prism with specific face areas and a dimension relation -/
structure RectangularPrism where
  x : ℝ
  y : ℝ
  z : ℝ
  side_area : x * y = 24
  front_area : y * z = 15
  bottom_area : x * z = 8
  dimension_relation : z = 2 * x

/-- The volume of a rectangular prism is equal to 96 cubic inches -/
theorem prism_volume (p : RectangularPrism) : p.x * p.y * p.z = 96 := by
  sorry

end prism_volume_l3498_349846


namespace problem_solution_l3498_349811

-- Define proposition p
def p : Prop := ∀ (x a : ℝ), x^2 + a*x + a^2 ≥ 0

-- Define proposition q
def q : Prop := ∃ (x : ℕ), x > 0 ∧ 2*x^2 - 1 ≤ 0

-- Theorem to prove
theorem problem_solution :
  p ∧ ¬q ∧ (p ∨ q) :=
sorry

end problem_solution_l3498_349811


namespace adam_has_ten_apples_l3498_349812

def apples_problem (jackie_apples adam_more_apples : ℕ) : Prop :=
  let adam_apples := jackie_apples + adam_more_apples
  adam_apples = 10

theorem adam_has_ten_apples :
  apples_problem 2 8 := by sorry

end adam_has_ten_apples_l3498_349812


namespace parentheses_expression_l3498_349845

theorem parentheses_expression (a b : ℝ) : (3*b + a) * (3*b - a) = 9*b^2 - a^2 := by
  sorry

end parentheses_expression_l3498_349845


namespace union_equality_implies_a_greater_than_one_l3498_349822

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem union_equality_implies_a_greater_than_one (a : ℝ) :
  A ∪ B a = B a → a > 1 := by
  sorry

end union_equality_implies_a_greater_than_one_l3498_349822


namespace problem_statement_l3498_349832

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := by
  sorry

end problem_statement_l3498_349832


namespace sum_of_a_and_b_l3498_349831

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 3 * a + 5 * b = 47) 
  (eq2 : 7 * a + 2 * b = 52) : 
  a + b = 35 / 3 := by
sorry

end sum_of_a_and_b_l3498_349831


namespace subsets_and_sum_of_M_l3498_349872

def M : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem subsets_and_sum_of_M :
  (Finset.powerset M).card = 2^10 ∧
  (Finset.powerset M).sum (λ s => s.sum id) = 55 * 2^9 := by
  sorry

end subsets_and_sum_of_M_l3498_349872


namespace geometric_sequence_ratio_l3498_349810

theorem geometric_sequence_ratio (q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  q = 1/2 →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, a (n + 1) = a n * q) →
  S 4 / a 3 = 15/2 := by
sorry

end geometric_sequence_ratio_l3498_349810


namespace committee_selection_count_l3498_349805

theorem committee_selection_count : Nat.choose 30 5 = 142506 := by sorry

end committee_selection_count_l3498_349805


namespace benny_picked_two_apples_l3498_349868

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The difference between the number of apples Dan and Benny picked -/
def difference : ℕ := 7

/-- The number of apples Benny picked -/
def benny_apples : ℕ := dan_apples - difference

theorem benny_picked_two_apples : benny_apples = 2 := by
  sorry

end benny_picked_two_apples_l3498_349868


namespace roots_expression_l3498_349801

theorem roots_expression (p q α β γ δ : ℝ) : 
  (α^2 - p*α + 1 = 0) → 
  (β^2 - p*β + 1 = 0) → 
  (γ^2 - q*γ + 1 = 0) → 
  (δ^2 - q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = p^2 - q^2 := by
  sorry

end roots_expression_l3498_349801


namespace valid_coloring_iff_even_product_l3498_349850

/-- Represents a chessboard coloring where each small square not on the perimeter has exactly two sides colored. -/
def ValidColoring (m n : ℕ) := True  -- Placeholder definition

/-- Theorem stating that a valid coloring exists if and only if m * n is even -/
theorem valid_coloring_iff_even_product (m n : ℕ) :
  ValidColoring m n ↔ Even (m * n) :=
by sorry

end valid_coloring_iff_even_product_l3498_349850


namespace number_ratio_l3498_349806

theorem number_ratio (A B C : ℝ) : 
  A + B + C = 110 → A = 2 * B → B = 30 → C / A = 1 / 3 := by
  sorry

end number_ratio_l3498_349806


namespace austin_surfboard_length_l3498_349859

/-- Austin's surfing problem -/
theorem austin_surfboard_length 
  (H : ℝ) -- Austin's height
  (S : ℝ) -- Austin's surfboard length
  (highest_wave : 4 * H + 2 = 26) -- Highest wave is 2 feet higher than 4 times Austin's height
  (shortest_wave_height : H + 4 = S + 3) -- Shortest wave is 4 feet higher than Austin's height and 3 feet higher than surfboard length
  : S = 7 := by
  sorry


end austin_surfboard_length_l3498_349859


namespace largest_multiple_of_15_under_500_l3498_349817

theorem largest_multiple_of_15_under_500 : ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  (∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495) := by
  sorry

end largest_multiple_of_15_under_500_l3498_349817


namespace peace_numbers_examples_l3498_349887

/-- Two numbers are peace numbers about 3 if their sum is 3 -/
def PeaceNumbersAbout3 (a b : ℝ) : Prop := a + b = 3

theorem peace_numbers_examples :
  (PeaceNumbersAbout3 4 (-1)) ∧
  (∀ x : ℝ, PeaceNumbersAbout3 (8 - x) (-5 + x)) ∧
  (∀ x : ℝ, PeaceNumbersAbout3 (x^2 - 4*x - 1) (x^2 - 2*(x^2 - 2*x - 2))) ∧
  (∀ k : ℕ, (∃ x : ℕ, x > 0 ∧ PeaceNumbersAbout3 (k * x + 1) (x - 2)) ↔ (k = 1 ∨ k = 3)) :=
by sorry

end peace_numbers_examples_l3498_349887


namespace most_suitable_sampling_method_l3498_349877

/-- Represents the age groups in the population --/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | ExcludeOneElderlyThenStratified

/-- Represents the population composition --/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Determines if a sampling method is suitable for a given population and sample size --/
def isSuitableMethod (pop : Population) (sampleSize : Nat) (method : SamplingMethod) : Prop :=
  sorry

/-- The theorem stating that excluding one elderly person and then using stratified sampling
    is the most suitable method for the given population and sample size --/
theorem most_suitable_sampling_method
  (pop : Population)
  (h1 : pop.elderly = 28)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (sampleSize : Nat)
  (h4 : sampleSize = 36) :
  isSuitableMethod pop sampleSize SamplingMethod.ExcludeOneElderlyThenStratified ∧
  ∀ m : SamplingMethod,
    isSuitableMethod pop sampleSize m →
    m = SamplingMethod.ExcludeOneElderlyThenStratified :=
  sorry


end most_suitable_sampling_method_l3498_349877


namespace lamp_game_solvable_l3498_349883

/-- Represents a move in the lamp game -/
inductive Move
  | row (r : Nat) (start : Nat)
  | col (c : Nat) (start : Nat)

/-- The lamp game state -/
def LampGame (n m : Nat) :=
  { grid : Fin n → Fin n → Bool // n > 0 ∧ m > 0 }

/-- Applies a move to the game state -/
def applyMove (game : LampGame n m) (move : Move) : LampGame n m :=
  sorry

/-- Checks if all lamps are on -/
def allOn (game : LampGame n m) : Prop :=
  ∀ i j, game.val i j = true

/-- Main theorem: all lamps can be turned on iff m divides n -/
theorem lamp_game_solvable (n m : Nat) :
  (∃ (game : LampGame n m) (moves : List Move), allOn (moves.foldl applyMove game)) ↔ m ∣ n :=
  sorry

end lamp_game_solvable_l3498_349883


namespace cubic_root_of_unity_solutions_l3498_349874

theorem cubic_root_of_unity_solutions (p q r s : ℂ) (m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (p * m^2 + q * m + r = 0) ∧ (q * m^2 + r * m + s = 0) →
  (m = 1) ∨ (m = Complex.exp ((2 * Real.pi * Complex.I) / 3)) ∨ (m = Complex.exp ((-2 * Real.pi * Complex.I) / 3)) :=
by sorry

end cubic_root_of_unity_solutions_l3498_349874


namespace train_speed_l3498_349884

/-- Proves that a train of given length crossing a platform of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 230 ∧ 
  platform_length = 290 ∧ 
  crossing_time = 26 →
  (train_length + platform_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed

end train_speed_l3498_349884


namespace pens_to_sell_for_profit_l3498_349865

theorem pens_to_sell_for_profit (total_pens : ℕ) (cost_per_pen sell_price : ℚ) (desired_profit : ℚ) :
  total_pens = 2000 →
  cost_per_pen = 15/100 →
  sell_price = 30/100 →
  desired_profit = 150 →
  ∃ (pens_to_sell : ℕ), 
    pens_to_sell * sell_price - total_pens * cost_per_pen = desired_profit ∧
    pens_to_sell = 1500 :=
by sorry

end pens_to_sell_for_profit_l3498_349865


namespace sufficient_not_necessary_l3498_349880

theorem sufficient_not_necessary (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0 → (x + 3) * (y - 4) = 0) ∧
  ¬((x + 3) * (y - 4) = 0 → (x + 3)^2 + (y - 4)^2 = 0) := by
  sorry

end sufficient_not_necessary_l3498_349880


namespace jude_bottle_cap_trading_l3498_349861

/-- Jude's bottle cap trading problem -/
theorem jude_bottle_cap_trading
  (initial_caps : ℕ)
  (car_cost : ℕ)
  (truck_cost : ℕ)
  (trucks_bought : ℕ)
  (total_vehicles : ℕ)
  (h1 : initial_caps = 100)
  (h2 : car_cost = 5)
  (h3 : truck_cost = 6)
  (h4 : trucks_bought = 10)
  (h5 : total_vehicles = 16) :
  (car_cost * (total_vehicles - trucks_bought) : ℚ) / (initial_caps - truck_cost * trucks_bought) = 3/4 := by
  sorry


end jude_bottle_cap_trading_l3498_349861


namespace circle_equation_from_diameter_endpoints_l3498_349836

theorem circle_equation_from_diameter_endpoints (x y : ℝ) :
  let p₁ : ℝ × ℝ := (0, 0)
  let p₂ : ℝ × ℝ := (6, 8)
  let center : ℝ × ℝ := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  let radius : ℝ := Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by
  sorry

end circle_equation_from_diameter_endpoints_l3498_349836


namespace slab_rate_calculation_l3498_349870

/-- Given a room with specific dimensions and total flooring cost, 
    prove that the rate per square meter for slabs is as calculated. -/
theorem slab_rate_calculation (length width total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : width = 3.75)
    (h3 : total_cost = 12375) : 
  total_cost / (length * width) = 600 := by
  sorry

end slab_rate_calculation_l3498_349870


namespace circle_chord_segments_l3498_349890

theorem circle_chord_segments (r : ℝ) (chord_length : ℝ) : 
  r = 6 → 
  chord_length = 10 → 
  ∃ (m n : ℝ), m + n = 2*r ∧ m*n = (chord_length/2)^2 ∧ 
  ((m = 6 + Real.sqrt 11 ∧ n = 6 - Real.sqrt 11) ∨ 
   (m = 6 - Real.sqrt 11 ∧ n = 6 + Real.sqrt 11)) :=
by sorry

end circle_chord_segments_l3498_349890


namespace james_money_theorem_l3498_349800

/-- The amount of money James has after finding some bills -/
def jamesTotal (billsFound : ℕ) (billValue : ℕ) (walletAmount : ℕ) : ℕ :=
  billsFound * billValue + walletAmount

/-- Theorem stating that James has $135 after finding 3 $20 bills -/
theorem james_money_theorem :
  jamesTotal 3 20 75 = 135 := by
  sorry

end james_money_theorem_l3498_349800


namespace hypotenuse_length_l3498_349898

-- Define a right triangle with legs 3 and 5
def right_triangle (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 5 ∧ c^2 = a^2 + b^2

-- Theorem statement
theorem hypotenuse_length :
  ∀ a b c : ℝ, right_triangle a b c → c = Real.sqrt 34 := by
  sorry

end hypotenuse_length_l3498_349898


namespace image_and_preimage_of_f_l3498_349840

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem image_and_preimage_of_f :
  (f (3, 5) = (-2, 8)) ∧ (f (4, 1) = (-2, 8)) := by sorry

end image_and_preimage_of_f_l3498_349840


namespace birds_on_fence_l3498_349844

/-- The total number of birds on a fence given initial birds, additional birds, and additional storks -/
def total_birds (initial : ℕ) (additional : ℕ) (storks : ℕ) : ℕ :=
  initial + additional + storks

/-- Theorem stating that given 6 initial birds, 4 additional birds, and 8 additional storks, 
    the total number of birds on the fence is 18 -/
theorem birds_on_fence : total_birds 6 4 8 = 18 := by
  sorry

end birds_on_fence_l3498_349844


namespace union_A_complement_B_l3498_349819

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3}

-- Define set B
def B : Set ℕ := {2, 3, 4}

-- Theorem to prove
theorem union_A_complement_B : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end union_A_complement_B_l3498_349819


namespace double_windows_count_l3498_349825

/-- Represents the number of glass panels in each window -/
def panels_per_window : ℕ := 4

/-- Represents the number of single windows upstairs -/
def single_windows : ℕ := 8

/-- Represents the total number of glass panels in the house -/
def total_panels : ℕ := 80

/-- Represents the number of double windows downstairs -/
def double_windows : ℕ := 12

/-- Theorem stating that the number of double windows downstairs is 12 -/
theorem double_windows_count : 
  panels_per_window * double_windows + panels_per_window * single_windows = total_panels :=
by sorry

end double_windows_count_l3498_349825


namespace hockey_goals_difference_l3498_349828

theorem hockey_goals_difference (layla_goals kristin_goals : ℕ) : 
  layla_goals = 104 →
  kristin_goals < layla_goals →
  (layla_goals + kristin_goals) / 2 = 92 →
  layla_goals - kristin_goals = 24 := by
sorry

end hockey_goals_difference_l3498_349828


namespace number_division_problem_l3498_349858

theorem number_division_problem (x : ℝ) : (x / 5 = 70 + x / 6) → x = 2100 := by
  sorry

end number_division_problem_l3498_349858
