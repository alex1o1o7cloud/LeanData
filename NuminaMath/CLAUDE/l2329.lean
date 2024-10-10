import Mathlib

namespace complex_fraction_simplification_l2329_232970

theorem complex_fraction_simplification :
  (7 : ℂ) + 15 * Complex.I / ((3 : ℂ) - 4 * Complex.I) = -39 / 25 + (73 / 25 : ℝ) * Complex.I :=
by sorry

end complex_fraction_simplification_l2329_232970


namespace parabola_f_value_l2329_232902

/-- Represents a parabola of the form x = dy² + ey + f -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.d * y^2 + p.e * y + p.f

theorem parabola_f_value (p : Parabola) :
  (p.x_coord 1 = -3) →  -- vertex at (-3, 1)
  (p.x_coord 3 = -1) →  -- passes through (-1, 3)
  (p.x_coord 0 = -2.5) →  -- passes through (-2.5, 0)
  p.f = -2.5 := by
  sorry

#check parabola_f_value

end parabola_f_value_l2329_232902


namespace each_brother_pays_19_80_l2329_232968

/-- The amount each brother pays when buying cakes and splitting the cost -/
def amount_per_person (num_cakes : ℕ) (price_per_cake : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_before_tax := num_cakes * price_per_cake
  let tax_amount := total_before_tax * tax_rate
  let total_after_tax := total_before_tax + tax_amount
  total_after_tax / 2

/-- Theorem stating that each brother pays $19.80 -/
theorem each_brother_pays_19_80 :
  amount_per_person 3 12 (1/10) = 198/10 := by
  sorry

end each_brother_pays_19_80_l2329_232968


namespace election_votes_calculation_l2329_232929

theorem election_votes_calculation (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.55 = (total_votes : ℝ) * 0.35 + 400 →
  total_votes = 2000 := by
sorry

end election_votes_calculation_l2329_232929


namespace calculation_proof_l2329_232999

theorem calculation_proof :
  ((-1/2) * (-8) + (-6) = -2) ∧
  (-1^4 - 2 / (-1/3) - |-9| = -4) := by
sorry

end calculation_proof_l2329_232999


namespace abc_product_magnitude_l2329_232952

theorem abc_product_magnitude (a b c : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a ≠ b → b ≠ c → c ≠ a →
  (a + 1 / b^2 = b + 1 / c^2) →
  (b + 1 / c^2 = c + 1 / a^2) →
  |a * b * c| = 1 := by
  sorry

end abc_product_magnitude_l2329_232952


namespace absolute_value_inequality_solution_set_l2329_232917

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 2| + |x - 2| ≤ 4} = Set.Icc (-2) 2 := by
  sorry

end absolute_value_inequality_solution_set_l2329_232917


namespace tangent_line_perpendicular_l2329_232920

theorem tangent_line_perpendicular (a : ℝ) : 
  let f (x : ℝ) := Real.exp (2 * a * x)
  let f' (x : ℝ) := 2 * a * Real.exp (2 * a * x)
  let tangent_slope := f' 0
  let perpendicular_line_slope := -1 / 2
  (tangent_slope = perpendicular_line_slope) → a = -1/4 := by
sorry

end tangent_line_perpendicular_l2329_232920


namespace smoking_health_correlation_l2329_232960

-- Define smoking and health as variables
variable (smoking health : ℝ)

-- Define the concept of "harmful to health"
def is_harmful_to_health (x y : ℝ) : Prop := 
  ∀ δ > 0, ∃ ε > 0, ∀ x' y', |x' - x| < ε → |y' - y| < δ → y' < y

-- Define negative correlation
def negative_correlation (x y : ℝ) : Prop :=
  ∀ δ > 0, ∃ ε > 0, ∀ x₁ x₂ y₁ y₂, 
    |x₁ - x| < ε → |x₂ - x| < ε → |y₁ - y| < δ → |y₂ - y| < δ →
    (x₁ < x₂ → y₁ > y₂) ∧ (x₁ > x₂ → y₁ < y₂)

-- Theorem statement
theorem smoking_health_correlation 
  (h : is_harmful_to_health smoking health) : 
  negative_correlation smoking health :=
sorry

end smoking_health_correlation_l2329_232960


namespace ax5_plus_by5_l2329_232959

theorem ax5_plus_by5 (a b x y : ℝ) 
  (eq1 : a*x + b*y = 1)
  (eq2 : a*x^2 + b*y^2 = 2)
  (eq3 : a*x^3 + b*y^3 = 5)
  (eq4 : a*x^4 + b*y^4 = 15) :
  a*x^5 + b*y^5 = -40 := by
  sorry

end ax5_plus_by5_l2329_232959


namespace four_tuple_count_l2329_232987

theorem four_tuple_count (p : ℕ) (hp : Prime p) : 
  (Finset.filter 
    (fun (t : ℕ × ℕ × ℕ × ℕ) => 
      0 < t.1 ∧ t.1 < p - 1 ∧
      0 < t.2.1 ∧ t.2.1 < p - 1 ∧
      0 < t.2.2.1 ∧ t.2.2.1 < p - 1 ∧
      0 < t.2.2.2 ∧ t.2.2.2 < p - 1 ∧
      (t.1 * t.2.2.2) % p = (t.2.1 * t.2.2.1) % p)
    (Finset.product 
      (Finset.range (p - 1)) 
      (Finset.product 
        (Finset.range (p - 1)) 
        (Finset.product 
          (Finset.range (p - 1)) 
          (Finset.range (p - 1)))))).card = (p - 1)^3 :=
by sorry


end four_tuple_count_l2329_232987


namespace class_average_age_l2329_232944

theorem class_average_age (initial_students : ℕ) (leaving_student_age : ℕ) (teacher_age : ℕ) (new_average : ℝ) :
  initial_students = 30 →
  leaving_student_age = 11 →
  teacher_age = 41 →
  new_average = 11 →
  (initial_students * (initial_average : ℝ) - leaving_student_age + teacher_age) / initial_students = new_average →
  initial_average = 10 :=
by
  sorry

end class_average_age_l2329_232944


namespace alternating_sequence_sum_l2329_232980

def sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  let last_term := a₁ + (n - 1) * d
  let sum_of_pairs := ((n - 1) / 2) * (a₁ + last_term - d)
  if n % 2 = 0 then sum_of_pairs else sum_of_pairs + last_term

theorem alternating_sequence_sum :
  sequence_sum 2 3 19 = 29 := by
  sorry

end alternating_sequence_sum_l2329_232980


namespace sum_of_fourth_powers_bound_l2329_232913

theorem sum_of_fourth_powers_bound (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 ≤ 1) :
  (a+b)^4 + (a+c)^4 + (a+d)^4 + (b+c)^4 + (b+d)^4 + (c+d)^4 ≤ 6 := by
  sorry

end sum_of_fourth_powers_bound_l2329_232913


namespace black_squares_count_l2329_232905

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard :=
  (size : Nat)
  (has_black_corners : Bool)
  (alternating : Bool)

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  sorry

/-- Theorem: A 32x32 checkerboard with black corners and alternating colors has 512 black squares -/
theorem black_squares_count (board : Checkerboard) :
  board.size = 32 ∧ board.has_black_corners ∧ board.alternating →
  count_black_squares board = 512 :=
sorry

end black_squares_count_l2329_232905


namespace unique_solution_xy_l2329_232937

/-- The unique solution to the system of equations x^y + 3 = y^x and 2x^y = y^x + 11 -/
theorem unique_solution_xy : ∃! (x y : ℕ+), 
  (x : ℝ) ^ (y : ℝ) + 3 = (y : ℝ) ^ (x : ℝ) ∧ 
  2 * (x : ℝ) ^ (y : ℝ) = (y : ℝ) ^ (x : ℝ) + 11 ∧
  x = 14 ∧ y = 1 := by
  sorry

end unique_solution_xy_l2329_232937


namespace R_symmetry_l2329_232977

/-- Recursive definition of R_n sequences -/
def R : ℕ → List ℕ
  | 0 => [1]
  | n + 1 =>
    let prev := R n
    List.join (prev.map (fun x => List.range x)) ++ [n + 1]

/-- Main theorem -/
theorem R_symmetry (n : ℕ) (k : ℕ) (h : n > 1) :
  (R n).nthLe k (by sorry) = 1 ↔
  (R n).nthLe ((R n).length - 1 - k) (by sorry) ≠ 1 :=
by sorry

end R_symmetry_l2329_232977


namespace age_ratio_theorem_l2329_232992

/-- Represents the ages of Arun and Deepak -/
structure Ages where
  arun : ℕ
  deepak : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of two natural numbers -/
def calculateRatio (a b : ℕ) : Ratio :=
  let gcd := Nat.gcd a b
  { numerator := a / gcd, denominator := b / gcd }

/-- Theorem stating the ratio of Arun's and Deepak's ages -/
theorem age_ratio_theorem (ages : Ages) : 
  ages.deepak = 42 → 
  ages.arun + 6 = 36 → 
  calculateRatio ages.arun ages.deepak = Ratio.mk 5 7 := by
  sorry

#check age_ratio_theorem

end age_ratio_theorem_l2329_232992


namespace division_of_fractions_l2329_232989

theorem division_of_fractions : (7 : ℚ) / (8 / 13) = 91 / 8 := by
  sorry

end division_of_fractions_l2329_232989


namespace opposite_of_negative_one_to_2023_l2329_232906

theorem opposite_of_negative_one_to_2023 :
  ∀ n : ℕ, n = 2023 → Odd n → (-((-1)^n)) = 1 := by
  sorry

end opposite_of_negative_one_to_2023_l2329_232906


namespace point_on_line_l2329_232953

/-- Given a line with equation x + 2y + 5 = 0, if (m, n) and (m + 2, n + k) are two points on this line,
    then k = -1 -/
theorem point_on_line (m n k : ℝ) : 
  (m + 2*n + 5 = 0) → ((m + 2) + 2*(n + k) + 5 = 0) → k = -1 := by
  sorry

end point_on_line_l2329_232953


namespace ellipse_chord_theorem_l2329_232973

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- A point bisects a chord of the ellipse -/
def bisects_chord (x y : ℝ) (px py : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    is_on_ellipse x1 y1 ∧
    is_on_ellipse x2 y2 ∧
    px = (x1 + x2) / 2 ∧
    py = (y1 + y2) / 2

/-- The equation of a line -/
def on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

theorem ellipse_chord_theorem :
  ∀ (x y : ℝ),
    is_on_ellipse x y →
    bisects_chord x y 4 2 →
    on_line x y 1 2 (-8) :=
by sorry

end ellipse_chord_theorem_l2329_232973


namespace farm_animals_count_l2329_232991

theorem farm_animals_count (rabbits chickens : ℕ) : 
  rabbits = chickens + 17 → 
  rabbits = 64 → 
  rabbits + chickens = 111 := by
sorry

end farm_animals_count_l2329_232991


namespace polynomial_monotonicity_l2329_232951

-- Define a polynomial function
variable (P : ℝ → ℝ)

-- Define strict monotonicity
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem polynomial_monotonicity
  (h1 : StrictlyMonotonic (λ x => P (P x)))
  (h2 : StrictlyMonotonic (λ x => P (P (P x))))
  : StrictlyMonotonic P := by
  sorry

end polynomial_monotonicity_l2329_232951


namespace two_numbers_problem_l2329_232936

theorem two_numbers_problem (a b : ℕ) :
  a + b = 667 →
  Nat.lcm a b / Nat.gcd a b = 120 →
  ((a = 115 ∧ b = 552) ∨ (a = 552 ∧ b = 115)) ∨
  ((a = 232 ∧ b = 435) ∨ (a = 435 ∧ b = 232)) := by
sorry

end two_numbers_problem_l2329_232936


namespace value_swap_l2329_232966

theorem value_swap (a b : ℕ) (h1 : a = 1) (h2 : b = 2) :
  let c := a
  let a' := b
  let b' := c
  (a', b', c) = (2, 1, 1) := by sorry

end value_swap_l2329_232966


namespace p_sufficient_not_necessary_for_q_l2329_232926

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 4
def q (x : ℝ) : Prop := x^2 - 5*x + 4 ≥ 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
sorry

end p_sufficient_not_necessary_for_q_l2329_232926


namespace min_value_expression_l2329_232950

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * x) / (3 * x + 2 * y) + y / (2 * x + y) ≥ 4 * Real.sqrt 3 - 6 := by
  sorry

end min_value_expression_l2329_232950


namespace probability_of_sum_five_l2329_232912

def number_of_faces : ℕ := 6

def total_outcomes (n : ℕ) : ℕ := n * n

def favorable_outcomes : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 2), (4, 1)]

def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

theorem probability_of_sum_five :
  probability (favorable_outcomes.length) (total_outcomes number_of_faces) = 1 / 9 := by
  sorry

end probability_of_sum_five_l2329_232912


namespace sum_of_reciprocals_l2329_232978

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end sum_of_reciprocals_l2329_232978


namespace total_fuel_needed_l2329_232930

def fuel_consumption : ℝ := 5
def trip1_distance : ℝ := 30
def trip2_distance : ℝ := 20

theorem total_fuel_needed : 
  fuel_consumption * (trip1_distance + trip2_distance) = 250 := by
sorry

end total_fuel_needed_l2329_232930


namespace tax_free_items_cost_l2329_232990

/-- Given a purchase with total cost, sales tax, and tax rate, calculate the cost of tax-free items -/
theorem tax_free_items_cost
  (total_cost : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_cost = 40)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.06)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = total_cost - sales_tax / tax_rate :=
by
  sorry

end tax_free_items_cost_l2329_232990


namespace consecutive_integers_base_sum_l2329_232915

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- Checks if a number is a valid base -/
def isValidBase (b : Nat) : Prop := b ≥ 2

theorem consecutive_integers_base_sum (C D : Nat) : 
  C.succ = D →
  isValidBase C →
  isValidBase D →
  isValidBase (C + D) →
  toBase10 [1, 4, 5] C + toBase10 [5, 6] D = toBase10 [9, 2] (C + D) →
  C + D = 11 := by
  sorry

#check consecutive_integers_base_sum

end consecutive_integers_base_sum_l2329_232915


namespace planes_count_theorem_l2329_232958

/-- A straight line in 3D space -/
structure Line3D where
  -- Define necessary properties for a line

/-- A point in 3D space -/
structure Point3D where
  -- Define necessary properties for a point

/-- A plane in 3D space -/
structure Plane3D where
  -- Define necessary properties for a plane

/-- Predicate to check if a point is outside a line -/
def is_outside (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if three points are collinear -/
def are_collinear (p1 p2 p3 : Point3D) : Prop :=
  sorry

/-- Function to count the number of unique planes determined by a line and three points -/
def count_planes (l : Line3D) (p1 p2 p3 : Point3D) : Nat :=
  sorry

/-- Theorem stating the possible number of planes -/
theorem planes_count_theorem (l : Line3D) (A B C : Point3D) 
  (h1 : is_outside A l)
  (h2 : is_outside B l)
  (h3 : is_outside C l) :
  (count_planes l A B C = 1) ∨ (count_planes l A B C = 3) ∨ (count_planes l A B C = 4) :=
by
  sorry

end planes_count_theorem_l2329_232958


namespace function_and_triangle_properties_l2329_232948

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.cos (ω * x) ^ 2 - 1/2

theorem function_and_triangle_properties 
  (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_distance : ∀ x₁ x₂, f ω x₁ = f ω x₂ → x₂ - x₁ = π / ω ∨ x₂ - x₁ = -π / ω) 
  (A B C : ℝ) 
  (h_c : Real.sqrt 7 = 2 * Real.sin (A/2) * Real.sin (B/2))
  (h_fC : f ω C = 0) 
  (h_sinB : Real.sin B = 3 * Real.sin A) :
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-π/6 + k*π) (k*π + π/3), 
    ∀ y ∈ Set.Icc (-π/6 + k*π) (k*π + π/3), 
    x ≤ y → f ω x ≤ f ω y) ∧
  2 * Real.sin (A/2) = 1 ∧ 
  2 * Real.sin (B/2) = 3 := by
sorry

end function_and_triangle_properties_l2329_232948


namespace parallel_vectors_x_value_l2329_232986

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x, 4)
  vector_parallel a b → x = 6 :=
by
  sorry

end parallel_vectors_x_value_l2329_232986


namespace grazing_months_a_l2329_232997

/-- The number of months a put his oxen for grazing -/
def months_a : ℕ := 7

/-- The number of oxen a put for grazing -/
def oxen_a : ℕ := 10

/-- The number of oxen b put for grazing -/
def oxen_b : ℕ := 12

/-- The number of months b put his oxen for grazing -/
def months_b : ℕ := 5

/-- The number of oxen c put for grazing -/
def oxen_c : ℕ := 15

/-- The number of months c put his oxen for grazing -/
def months_c : ℕ := 3

/-- The total rent of the pasture in rupees -/
def total_rent : ℚ := 245

/-- The share of rent c pays in rupees -/
def c_rent_share : ℚ := 62.99999999999999

theorem grazing_months_a : 
  months_a * oxen_a * total_rent = 
  c_rent_share * (months_a * oxen_a + months_b * oxen_b + months_c * oxen_c) := by
  sorry

end grazing_months_a_l2329_232997


namespace range_of_f_on_interval_l2329_232940

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 16 * x + 1

-- State the theorem
theorem range_of_f_on_interval :
  let a := 1
  let b := 2
  (∀ x ≤ -2, ∀ y ∈ Set.Ioo x (-2), f x ≥ f y) →
  (∀ x ≥ -2, ∀ y ∈ Set.Ioo (-2) x, f x ≥ f y) →
  Set.range (fun x ↦ f x) ∩ Set.Icc a b = Set.Icc (f a) (f b) :=
by sorry

end range_of_f_on_interval_l2329_232940


namespace certain_number_proof_l2329_232996

theorem certain_number_proof (x : ℝ) (h : x = 3) :
  ∃ y : ℝ, (x + y) / (x + y + 5) = (x + y + 5) / (x + y + 5 + 13) ∧ y = 1/8 := by
  sorry

end certain_number_proof_l2329_232996


namespace negation_of_proposition_is_true_l2329_232961

theorem negation_of_proposition_is_true : 
  (∃ a : ℝ, a > 2 ∧ a^2 ≥ 4) := by sorry

end negation_of_proposition_is_true_l2329_232961


namespace probability_three_suits_standard_deck_l2329_232955

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- The probability of drawing three cards from a standard deck
    and getting one each from ♠, ♥, and ♦ suits (in any order) -/
def probability_three_suits (d : Deck) : ℚ :=
  let total_outcomes := d.cards * (d.cards - 1) * (d.cards - 2)
  let favorable_outcomes := d.ranks * d.ranks * d.ranks * 6
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability for a standard 52-card deck -/
theorem probability_three_suits_standard_deck :
  probability_three_suits ⟨52, 13, 4⟩ = 2197 / 22100 := by
  sorry

#eval probability_three_suits ⟨52, 13, 4⟩

end probability_three_suits_standard_deck_l2329_232955


namespace perfect_square_triples_l2329_232993

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem perfect_square_triples :
  ∀ a b c : ℕ,
    (is_perfect_square (a^2 + 2*b + c) ∧
     is_perfect_square (b^2 + 2*c + a) ∧
     is_perfect_square (c^2 + 2*a + b)) →
    ((a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 43 ∧ b = 127 ∧ c = 106)) :=
by sorry

end perfect_square_triples_l2329_232993


namespace tim_initial_books_l2329_232994

/-- The number of books Sandy has -/
def sandy_books : ℕ := 10

/-- The number of books Benny lost -/
def benny_lost : ℕ := 24

/-- The number of books they have together after Benny lost some -/
def remaining_books : ℕ := 19

/-- Tim's initial number of books -/
def tim_books : ℕ := 33

theorem tim_initial_books : 
  sandy_books + tim_books - benny_lost = remaining_books :=
by sorry

end tim_initial_books_l2329_232994


namespace water_tank_capacity_l2329_232956

theorem water_tank_capacity (tank_capacity : ℝ) : 
  (0.6 * tank_capacity - (0.7 * tank_capacity) = 45) → 
  tank_capacity = 450 := by
  sorry

end water_tank_capacity_l2329_232956


namespace arithmetic_sequence_ninth_term_l2329_232981

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 9th term of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_sum : a 4 + a 6 = 8) :
  a 9 = 5 := by
sorry

end arithmetic_sequence_ninth_term_l2329_232981


namespace rational_root_condition_l2329_232941

theorem rational_root_condition (n : ℕ+) :
  (∃ (x : ℚ), x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end rational_root_condition_l2329_232941


namespace a_arithmetic_l2329_232925

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

def q : ℝ := sorry

axiom q_neq_zero_one : q * (q - 1) ≠ 0

axiom sum_relation (n : ℕ) : (1 - q) * S n + q * a n = 1

axiom S_arithmetic : S 3 - S 9 = S 9 - S 6

theorem a_arithmetic : a 2 - a 8 = a 8 - a 5 := by sorry

end a_arithmetic_l2329_232925


namespace equation_solutions_l2329_232995

def is_solution (m n r k : ℕ+) : Prop :=
  m * n + n * r + m * r = k * (m + n + r)

theorem equation_solutions :
  (∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = 7 ∧ 
    (∀ x ∈ s, is_solution x.1 x.2.1 x.2.2 2) ∧
    (∀ x : ℕ+ × ℕ+ × ℕ+, is_solution x.1 x.2.1 x.2.2 2 → x ∈ s)) ∧
  (∀ k : ℕ+, k > 1 → 
    ∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card ≥ 3 * k + 1 ∧ 
      ∀ x ∈ s, is_solution x.1 x.2.1 x.2.2 k) :=
by sorry

end equation_solutions_l2329_232995


namespace determinant_k_value_l2329_232922

def determinant (a b c d e f g h i : ℝ) : ℝ :=
  a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h

def algebraic_cofactor_1_2 (a b c d e f g h i : ℝ) : ℝ :=
  -(b * i - c * h)

theorem determinant_k_value (k : ℝ) :
  algebraic_cofactor_1_2 4 2 k (-3) 5 4 (-1) 1 (-2) = -10 →
  k = -14 := by
  sorry

end determinant_k_value_l2329_232922


namespace jerrys_coins_l2329_232963

theorem jerrys_coins (n d : ℕ) : 
  n + d = 30 →
  5 * n + 10 * d + 140 = 10 * n + 5 * d →
  5 * n + 10 * d = 155 :=
by sorry

end jerrys_coins_l2329_232963


namespace valid_a_values_l2329_232918

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1 ∧ a ≥ 0}

def full_eating (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def partial_eating (X Y : Set ℝ) : Prop :=
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

def valid_relationship (a : ℝ) : Prop :=
  full_eating A (B a) ∨ partial_eating A (B a)

theorem valid_a_values : {a : ℝ | valid_relationship a} = {0, 1, 4} := by sorry

end valid_a_values_l2329_232918


namespace event_probability_l2329_232998

theorem event_probability (n : ℕ) (k₀ : ℕ) (p : ℝ) 
  (h1 : n = 120) 
  (h2 : k₀ = 32) 
  (h3 : k₀ = Int.floor (n * p)) :
  32 / 121 ≤ p ∧ p ≤ 33 / 121 := by
  sorry

end event_probability_l2329_232998


namespace prism_path_lengths_l2329_232938

/-- Regular triangular prism with given properties -/
structure RegularTriangularPrism where
  -- Base edge length
  ab : ℝ
  -- Height
  aa1 : ℝ
  -- Point on base edge BC
  p : ℝ × ℝ × ℝ
  -- Shortest path length from P to M
  shortest_path : ℝ

/-- Theorem stating the lengths of PC and NC in the given prism -/
theorem prism_path_lengths (prism : RegularTriangularPrism)
  (h_ab : prism.ab = 3)
  (h_aa1 : prism.aa1 = 4)
  (h_path : prism.shortest_path = Real.sqrt 29) :
  ∃ (pc nc : ℝ), pc = 2 ∧ nc = 4/5 := by
  sorry

end prism_path_lengths_l2329_232938


namespace seashell_points_sum_l2329_232935

/-- The total points earned for seashells collected by Joan, Jessica, and Jeremy -/
def total_points (joan_shells : ℕ) (joan_points : ℕ) (jessica_shells : ℕ) (jessica_points : ℕ) (jeremy_shells : ℕ) (jeremy_points : ℕ) : ℕ :=
  joan_shells * joan_points + jessica_shells * jessica_points + jeremy_shells * jeremy_points

/-- Theorem stating that the total points earned is 48 -/
theorem seashell_points_sum :
  total_points 6 2 8 3 12 1 = 48 := by
  sorry

end seashell_points_sum_l2329_232935


namespace sum_of_prime_factors_l2329_232923

def P (x : ℕ) : ℕ := x^6 + x^5 + x^3 + 1

theorem sum_of_prime_factors (h1 : 23 ∣ 67208001) (h2 : P 20 = 67208001) :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (67208001 + 1))) id) = 781 :=
sorry

end sum_of_prime_factors_l2329_232923


namespace units_digit_17_31_l2329_232910

theorem units_digit_17_31 : (17^31) % 10 = 3 := by
  sorry

end units_digit_17_31_l2329_232910


namespace min_distance_parabola_to_line_l2329_232982

/-- The minimum distance from a point on the parabola y = x^2 + 1 to the line y = 2x - 1 is √5/5 -/
theorem min_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2 + 1}
  let line := {p : ℝ × ℝ | p.2 = 2 * p.1 - 1}
  (∀ p ∈ parabola, ∃ q ∈ line, ∀ r ∈ line, dist p q ≤ dist p r) →
  (∃ p ∈ parabola, ∃ q ∈ line, dist p q = Real.sqrt 5 / 5) ∧
  (∀ p ∈ parabola, ∀ q ∈ line, dist p q ≥ Real.sqrt 5 / 5) :=
by sorry


end min_distance_parabola_to_line_l2329_232982


namespace movie_change_theorem_l2329_232975

/-- The change received by two sisters after buying movie tickets -/
def change_received (ticket_price : ℕ) (money_brought : ℕ) : ℕ :=
  money_brought - (2 * ticket_price)

/-- Theorem: The change received is $9 when tickets cost $8 each and the sisters brought $25 -/
theorem movie_change_theorem : change_received 8 25 = 9 := by
  sorry

end movie_change_theorem_l2329_232975


namespace lucy_fish_count_l2329_232916

-- Define the given quantities
def current_fish : ℕ := 212
def additional_fish : ℕ := 68

-- Define the total fish Lucy wants to have
def total_fish : ℕ := current_fish + additional_fish

-- Theorem statement
theorem lucy_fish_count : total_fish = 280 := by
  sorry

end lucy_fish_count_l2329_232916


namespace frank_work_hours_l2329_232947

/-- The number of hours Frank worked per day -/
def hours_per_day : ℕ := 8

/-- The number of days Frank worked -/
def days_worked : ℕ := 4

/-- The total number of hours Frank worked -/
def total_hours : ℕ := hours_per_day * days_worked

theorem frank_work_hours : total_hours = 32 := by
  sorry

end frank_work_hours_l2329_232947


namespace vacation_cost_l2329_232939

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 4 = 60) → C = 720 := by
sorry

end vacation_cost_l2329_232939


namespace rectangular_region_ratio_l2329_232976

theorem rectangular_region_ratio (L W : ℝ) (k : ℝ) : 
  L > 0 → W > 0 → k > 0 →
  L = k * W →
  L * W = 200 →
  2 * W + L = 40 →
  L / W = 2 := by
sorry

end rectangular_region_ratio_l2329_232976


namespace sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l2329_232967

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let s₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let s₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → s₁^2 + s₂^2 = (b^2 - 2*a*c) / a^2 := by
  sorry

theorem sum_of_squares_specific_quadratic :
  let s₁ := (15 + Real.sqrt 201) / 2
  let s₂ := (15 - Real.sqrt 201) / 2
  x^2 - 15*x + 6 = 0 → s₁^2 + s₂^2 = 213 := by
  sorry

end sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l2329_232967


namespace some_seniors_not_club_members_l2329_232934

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Senior : U → Prop)
variable (Punctual : U → Prop)
variable (ClubMember : U → Prop)

-- State the theorem
theorem some_seniors_not_club_members
  (h1 : ∃ x, Senior x ∧ ¬Punctual x)
  (h2 : ∀ x, ClubMember x → Punctual x) :
  ∃ x, Senior x ∧ ¬ClubMember x :=
by
  sorry


end some_seniors_not_club_members_l2329_232934


namespace min_value_of_expression_l2329_232979

theorem min_value_of_expression (x : ℝ) :
  let f := λ x : ℝ => Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((x - 1)^2 + (x - 1)^2)
  (∀ x, f x ≥ 1) ∧ (∃ x, f x = 1) := by sorry

end min_value_of_expression_l2329_232979


namespace power_calculation_l2329_232924

theorem power_calculation : (-1/2 : ℚ)^2023 * 2^2022 = -1/2 := by
  sorry

end power_calculation_l2329_232924


namespace tom_allowance_l2329_232900

theorem tom_allowance (initial_allowance : ℝ) 
  (first_week_fraction : ℝ) (second_week_fraction : ℝ) : 
  initial_allowance = 12 →
  first_week_fraction = 1/3 →
  second_week_fraction = 1/4 →
  let remaining_after_first_week := initial_allowance - (initial_allowance * first_week_fraction)
  let final_remaining := remaining_after_first_week - (remaining_after_first_week * second_week_fraction)
  final_remaining = 6 := by
sorry

end tom_allowance_l2329_232900


namespace rational_results_l2329_232983

-- Define the natural logarithm (ln) and common logarithm (lg)
noncomputable def ln (x : ℝ) := Real.log x
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the logarithm with an arbitrary base
noncomputable def log (b : ℝ) (x : ℝ) := ln x / ln b

-- State the theorem
theorem rational_results :
  (2 * lg 2 + lg 25 = 2) ∧
  (3^(1 / ln 3) - Real.exp 1 = 0) ∧
  (log 4 3 * log 3 6 * log 6 8 = 3/2) := by sorry

end rational_results_l2329_232983


namespace polynomial_division_theorem_l2329_232908

theorem polynomial_division_theorem (x : ℝ) :
  8 * x^3 - 2 * x^2 + 4 * x - 7 = (x - 1) * (8 * x^2 + 6 * x + 10) + 3 := by
  sorry

end polynomial_division_theorem_l2329_232908


namespace perpendicular_vectors_x_value_l2329_232971

theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![(-2), 1]
  let b : Fin 2 → ℝ := ![x, 2]
  (∀ i : Fin 2, a i * b i = 0) →
  x = 1 :=
by
  sorry

end perpendicular_vectors_x_value_l2329_232971


namespace focus_of_our_parabola_l2329_232911

/-- A parabola is defined by the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k + 1/(4a)) where (h, k) is the vertex -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola :=
  { a := 4
    b := 8
    c := -1 }

theorem focus_of_our_parabola :
  focus our_parabola = (-1, -79/16) := by sorry

end focus_of_our_parabola_l2329_232911


namespace race_result_l2329_232942

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ → ℝ

/-- The race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  runner_c : Runner

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.distance = 100 ∧
  r.runner_a.speed > r.runner_b.speed ∧
  r.runner_b.speed > r.runner_c.speed ∧
  (∀ t, r.runner_a.position t = r.runner_a.speed * t) ∧
  (∀ t, r.runner_b.position t = r.runner_b.speed * t) ∧
  (∀ t, r.runner_c.position t = r.runner_c.speed * t) ∧
  (∃ t_a, r.runner_a.position t_a = r.distance ∧ r.runner_b.position t_a = r.distance - 10) ∧
  (∃ t_b, r.runner_b.position t_b = r.distance ∧ r.runner_c.position t_b = r.distance - 10)

/-- The theorem to be proved -/
theorem race_result (r : Race) (h : race_conditions r) :
  ∃ t, r.runner_a.position t = r.distance ∧ r.runner_c.position t = r.distance - 19 := by
  sorry

end race_result_l2329_232942


namespace cosine_power_sum_l2329_232928

theorem cosine_power_sum (α : ℝ) (n : ℤ) (x : ℝ) (hx : x ≠ 0) :
  x + 1/x = 2 * Real.cos α →
  x^n + 1/x^n = 2 * Real.cos (n * α) := by
sorry

end cosine_power_sum_l2329_232928


namespace remove_six_for_average_l2329_232931

def original_list : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def removed_number : ℕ := 6

def remaining_list : List ℕ := original_list.filter (· ≠ removed_number)

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem remove_six_for_average :
  average remaining_list = 71/10 :=
sorry

end remove_six_for_average_l2329_232931


namespace total_smoothie_time_l2329_232949

/-- The time it takes to freeze ice cubes (in minutes) -/
def freezing_time : ℕ := 40

/-- The time it takes to make one smoothie (in minutes) -/
def smoothie_time : ℕ := 3

/-- The number of smoothies to be made -/
def num_smoothies : ℕ := 5

/-- Theorem stating the total time to make the smoothies -/
theorem total_smoothie_time : 
  freezing_time + num_smoothies * smoothie_time = 55 := by
  sorry

end total_smoothie_time_l2329_232949


namespace sum_of_three_squares_l2329_232945

theorem sum_of_three_squares (n : ℕ+) (h : ∃ m : ℕ, 3 * n + 1 = m^2) :
  ∃ a b c : ℕ, n + 1 = a^2 + b^2 + c^2 := by
sorry

end sum_of_three_squares_l2329_232945


namespace omicron_ba3_sample_size_l2329_232972

/-- The number of Omicron BA.3 virus strains in a stratified random sample -/
theorem omicron_ba3_sample_size 
  (total_strains : ℕ) 
  (ba3_strains : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_strains = 120) 
  (h2 : ba3_strains = 40) 
  (h3 : sample_size = 30) :
  (ba3_strains : ℚ) / total_strains * sample_size = 10 :=
sorry

end omicron_ba3_sample_size_l2329_232972


namespace KHSO4_moles_formed_l2329_232903

/-- Represents a chemical substance -/
inductive Substance
  | KOH
  | H2SO4
  | KHSO4
  | H2O

/-- Represents the balanced chemical equation -/
def balancedEquation : List (Nat × Substance) → List (Nat × Substance) → Prop :=
  fun reactants products =>
    reactants = [(1, Substance.KOH), (1, Substance.H2SO4)] ∧
    products = [(1, Substance.KHSO4), (1, Substance.H2O)]

/-- Theorem: The number of moles of KHSO4 formed is 2 -/
theorem KHSO4_moles_formed
  (koh_moles : Nat)
  (h2so4_moles : Nat)
  (h_koh : koh_moles = 2)
  (h_h2so4 : h2so4_moles = 2)
  (h_equation : balancedEquation [(1, Substance.KOH), (1, Substance.H2SO4)] [(1, Substance.KHSO4), (1, Substance.H2O)]) :
  (min koh_moles h2so4_moles) = 2 := by
  sorry

end KHSO4_moles_formed_l2329_232903


namespace wildlife_sanctuary_count_l2329_232919

theorem wildlife_sanctuary_count (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300) 
  (h2 : total_legs = 780) : ∃ (birds insects : ℕ),
  birds + insects = total_heads ∧
  2 * birds + 6 * insects = total_legs ∧
  birds = 255 := by
sorry

end wildlife_sanctuary_count_l2329_232919


namespace complement_of_union_M_N_l2329_232984

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem complement_of_union_M_N : 
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end complement_of_union_M_N_l2329_232984


namespace eccentricity_is_sqrt_five_l2329_232988

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- Represents a point on a hyperbola -/
structure PointOnHyperbola {a b : ℝ} (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The left and right foci of a hyperbola -/
def foci {a b : ℝ} (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The distance between a point and a focus -/
def dist_to_focus {a b : ℝ} (h : Hyperbola a b) (p : PointOnHyperbola h) (focus : ℝ) : ℝ := sorry

/-- The angle between the lines from a point to the foci -/
def angle_between_foci {a b : ℝ} (h : Hyperbola a b) (p : PointOnHyperbola h) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity {a b : ℝ} (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If there exists a point on the hyperbola where the angle between the lines to the foci is 90° and the distance to one focus is twice the distance to the other, then the eccentricity is √5 -/
theorem eccentricity_is_sqrt_five {a b : ℝ} (h : Hyperbola a b) :
  (∃ p : PointOnHyperbola h, 
    angle_between_foci h p = Real.pi / 2 ∧ 
    dist_to_focus h p (foci h).1 = 2 * dist_to_focus h p (foci h).2) →
  eccentricity h = Real.sqrt 5 := by sorry

end eccentricity_is_sqrt_five_l2329_232988


namespace right_triangle_third_side_l2329_232969

theorem right_triangle_third_side : ∀ a b c : ℝ,
  (a^2 - 9*a + 20 = 0) →
  (b^2 - 9*b + 20 = 0) →
  (a ≠ b) →
  (a^2 + b^2 = c^2) →
  (c = 3 ∨ c = Real.sqrt 41) :=
by sorry

end right_triangle_third_side_l2329_232969


namespace nuts_in_third_box_l2329_232927

-- Define the weights of nuts in each box
def box1 (x y z : ℝ) : ℝ := y + z - 6
def box2 (x y z : ℝ) : ℝ := x + z - 10

-- Theorem statement
theorem nuts_in_third_box (x y z : ℝ) 
  (h1 : x = box1 x y z) 
  (h2 : y = box2 x y z) : 
  z = 16 := by
sorry

end nuts_in_third_box_l2329_232927


namespace angle_C_is_30_degrees_l2329_232954

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  3 * Real.sin t.A + 4 * Real.cos t.B = 6

def condition2 (t : Triangle) : Prop :=
  4 * Real.sin t.B + 3 * Real.cos t.A = 1

-- Theorem statement
theorem angle_C_is_30_degrees (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.C = Real.pi / 6 := by sorry

end angle_C_is_30_degrees_l2329_232954


namespace snow_probability_in_ten_days_l2329_232914

/-- Probability of snow on a given day -/
def snow_prob (day : ℕ) : ℚ :=
  if day ≤ 5 then 1/5 else 1/3

/-- Probability of temperature dropping below 0°C -/
def cold_prob : ℚ := 1/2

/-- Increase in snow probability when temperature drops below 0°C -/
def snow_prob_increase : ℚ := 1/10

/-- Adjusted probability of no snow on a given day -/
def adj_no_snow_prob (day : ℕ) : ℚ :=
  cold_prob * (1 - snow_prob day) + (1 - cold_prob) * (1 - snow_prob day - snow_prob_increase)

/-- Probability of no snow for the entire period -/
def no_snow_prob : ℚ :=
  (adj_no_snow_prob 1)^5 * (adj_no_snow_prob 6)^5

theorem snow_probability_in_ten_days :
  1 - no_snow_prob = 58806/59049 :=
sorry

end snow_probability_in_ten_days_l2329_232914


namespace complement_A_intersect_B_l2329_232957

-- Define the set A
def A : Set ℝ := {x | x^2 - x ≤ 0}

-- Define the function f
def f (x : ℝ) : ℝ := 2 - x

-- Define the set B as the range of f on A
def B : Set ℝ := f '' A

-- State the theorem
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = Set.Ioo 1 2 := by sorry

end complement_A_intersect_B_l2329_232957


namespace f_of_three_equals_zero_l2329_232964

theorem f_of_three_equals_zero (f : ℝ → ℝ) (h : ∀ x, f (1 - 2*x) = x^2 + x) : f 3 = 0 := by
  sorry

end f_of_three_equals_zero_l2329_232964


namespace abs_negative_2023_l2329_232974

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_negative_2023_l2329_232974


namespace min_value_a_plus_8b_l2329_232965

theorem min_value_a_plus_8b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b = a + 2 * b) :
  ∀ x y, x > 0 → y > 0 → 2 * x * y = x + 2 * y → x + 8 * y ≥ 9 :=
sorry

end min_value_a_plus_8b_l2329_232965


namespace johnson_prescription_l2329_232907

/-- Represents a prescription with a fixed daily dose -/
structure Prescription where
  totalDays : ℕ
  remainingPills : ℕ
  daysElapsed : ℕ
  dailyDose : ℕ

/-- Calculates the daily dose given a prescription -/
def calculateDailyDose (p : Prescription) : ℕ :=
  (p.totalDays * p.dailyDose - p.remainingPills) / p.daysElapsed

/-- Theorem stating that for the given prescription, the daily dose is 2 pills -/
theorem johnson_prescription :
  ∃ (p : Prescription),
    p.totalDays = 30 ∧
    p.remainingPills = 12 ∧
    p.daysElapsed = 24 ∧
    calculateDailyDose p = 2 :=
by
  sorry


end johnson_prescription_l2329_232907


namespace profit_percentage_l2329_232943

/-- Given that the cost price of 150 articles equals the selling price of 120 articles,
    prove that the percent profit is 25%. -/
theorem profit_percentage (cost selling : ℝ) (h : 150 * cost = 120 * selling) :
  (selling - cost) / cost * 100 = 25 := by
  sorry

end profit_percentage_l2329_232943


namespace clock_hands_at_30_degrees_48_times_daily_l2329_232901

/-- Represents a clock with an hour hand and a minute hand -/
structure Clock where
  hour_hand : ℝ
  minute_hand : ℝ

/-- The speed of the minute hand relative to the hour hand -/
def minute_hand_speed : ℝ := 12

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The angle between clock hands we're interested in -/
def target_angle : ℝ := 30

/-- Function to count the number of times the clock hands form the target angle in a day -/
def count_target_angle_occurrences (c : Clock) : ℕ :=
  sorry

theorem clock_hands_at_30_degrees_48_times_daily :
  ∀ c : Clock, count_target_angle_occurrences c = 48 :=
sorry

end clock_hands_at_30_degrees_48_times_daily_l2329_232901


namespace running_distance_l2329_232962

theorem running_distance (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ) 
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ) 
  (h3 : davonte_distance = mercedes_distance + 2) :
  mercedes_distance + davonte_distance = 32 := by
sorry

end running_distance_l2329_232962


namespace sum_of_digits_in_period_l2329_232946

def period_length (n : ℕ) : ℕ := sorry

def decimal_expansion (n : ℕ) : List ℕ := sorry

theorem sum_of_digits_in_period (n : ℕ) (h : n = 98^2) :
  let m := period_length n
  let digits := decimal_expansion n
  List.sum (List.take m digits) = 900 := by sorry

end sum_of_digits_in_period_l2329_232946


namespace sixteen_is_counterexample_l2329_232985

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

def is_counterexample (n : Nat) : Prop :=
  ¬(is_prime n) ∧ (is_prime (n - 2) ∨ is_prime (n + 2))

theorem sixteen_is_counterexample : is_counterexample 16 := by
  sorry

end sixteen_is_counterexample_l2329_232985


namespace binomial_seven_two_minus_three_l2329_232904

theorem binomial_seven_two_minus_three : Nat.choose 7 2 - 3 = 18 := by
  sorry

end binomial_seven_two_minus_three_l2329_232904


namespace valid_numbering_exists_l2329_232921

/-- Represents a numbering system for 7 contacts and 7 holes -/
def Numbering := Fin 7 → Fin 7

/-- Checks if a numbering system satisfies the alignment condition for all rotations -/
def isValidNumbering (n : Numbering) : Prop :=
  ∀ k : Fin 7, ∃ i : Fin 7, n i = (i + k : Fin 7)

/-- The main theorem stating that a valid numbering system exists -/
theorem valid_numbering_exists : ∃ n : Numbering, isValidNumbering n := by
  sorry


end valid_numbering_exists_l2329_232921


namespace finite_rule2_applications_l2329_232932

/-- Represents the state of the blackboard -/
def Blackboard := List ℤ

/-- Rule 1: If there's a pair of equal numbers, add a to one and b to the other -/
def applyRule1 (board : Blackboard) (a b : ℕ) : Blackboard :=
  sorry

/-- Rule 2: If there's no pair of equal numbers, write two zeros -/
def applyRule2 : Blackboard := [0, 0]

/-- Applies either Rule 1 or Rule 2 based on the current board state -/
def applyRule (board : Blackboard) (a b : ℕ) : Blackboard :=
  sorry

/-- Represents a sequence of rule applications -/
def RuleSequence := List (Blackboard → Blackboard)

/-- Counts the number of times Rule 2 is applied in a sequence -/
def countRule2Applications (seq : RuleSequence) : ℕ :=
  sorry

/-- The main theorem: Rule 2 is applied only finitely many times -/
theorem finite_rule2_applications (a b : ℕ) (h : a ≠ b) :
  ∃ N : ℕ, ∀ seq : RuleSequence, countRule2Applications seq ≤ N :=
  sorry

end finite_rule2_applications_l2329_232932


namespace solution_set_is_ray_iff_l2329_232933

/-- The polynomial function representing the left side of the inequality -/
def f (a x : ℝ) : ℝ := x^3 - (a^2 + a + 1)*x^2 + (a^3 + a^2 + a)*x - a^3

/-- The set of solutions to the inequality for a given a -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- A set is a ray if it's of the form [c, ∞) or (-∞, c] for some c ∈ ℝ -/
def IsRay (S : Set ℝ) : Prop :=
  ∃ c : ℝ, S = {x : ℝ | x ≥ c} ∨ S = {x : ℝ | x ≤ c}

/-- The main theorem: The solution set is a ray iff a = 1 or a = -1 -/
theorem solution_set_is_ray_iff (a : ℝ) :
  IsRay (SolutionSet a) ↔ a = 1 ∨ a = -1 := by sorry

end solution_set_is_ray_iff_l2329_232933


namespace cube_root_simplification_l2329_232909

theorem cube_root_simplification :
  (8 + 27) ^ (1/3) * (8 + 27^(1/3)) ^ (1/3) = 385 ^ (1/3) := by
  sorry

end cube_root_simplification_l2329_232909
