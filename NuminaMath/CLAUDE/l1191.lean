import Mathlib

namespace NUMINAMATH_CALUDE_select_at_least_one_first_class_l1191_119174

theorem select_at_least_one_first_class :
  let total_parts : ℕ := 10
  let first_class_parts : ℕ := 6
  let second_class_parts : ℕ := 4
  let parts_to_select : ℕ := 3
  let total_combinations := Nat.choose total_parts parts_to_select
  let all_second_class := Nat.choose second_class_parts parts_to_select
  total_combinations - all_second_class = 116 := by
  sorry

end NUMINAMATH_CALUDE_select_at_least_one_first_class_l1191_119174


namespace NUMINAMATH_CALUDE_largest_band_size_l1191_119144

/-- Represents a band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid --/
def isValidFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  f.rows * f.membersPerRow + 3 = totalMembers

/-- Checks if the new formation after rearrangement is valid --/
def isValidNewFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  (f.rows - 3) * (f.membersPerRow + 2) = totalMembers

/-- Main theorem: The largest possible number of band members is 147 --/
theorem largest_band_size :
  ∃ (f : BandFormation) (m : ℕ),
    m < 150 ∧
    isValidFormation f m ∧
    isValidNewFormation f m ∧
    ∀ (f' : BandFormation) (m' : ℕ),
      m' < 150 →
      isValidFormation f' m' →
      isValidNewFormation f' m' →
      m' ≤ m ∧
    m = 147 := by
  sorry

end NUMINAMATH_CALUDE_largest_band_size_l1191_119144


namespace NUMINAMATH_CALUDE_farey_sequence_mediant_l1191_119168

theorem farey_sequence_mediant (r s : ℕ+) : 
  (6:ℚ)/11 < r/s ∧ r/s < (5:ℚ)/9 ∧ 
  (∀ (r' s' : ℕ+), (6:ℚ)/11 < r'/s' ∧ r'/s' < (5:ℚ)/9 → s ≤ s') →
  s - r = 9 :=
by sorry

end NUMINAMATH_CALUDE_farey_sequence_mediant_l1191_119168


namespace NUMINAMATH_CALUDE_coordinates_sum_of_X_l1191_119104

-- Define the points as pairs of real numbers
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := (3, 5)
def Z : ℝ × ℝ := (1, -3)

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem coordinates_sum_of_X :
  (distance X Z) / (distance X Y) = 1/2 ∧
  (distance Z Y) / (distance X Y) = 1/2 →
  X.1 + X.2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_X_l1191_119104


namespace NUMINAMATH_CALUDE_intersection_point_l1191_119162

-- Define the two linear functions
def f (x : ℝ) (m : ℝ) : ℝ := x + m
def g (x : ℝ) : ℝ := 2 * x - 2

-- Theorem statement
theorem intersection_point (m : ℝ) : 
  (∃ y : ℝ, f 0 m = y ∧ g 0 = y) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1191_119162


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1191_119116

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 12) 
  (eq2 : x + 4 * y = 18) : 
  17 * x^2 + 24 * x * y + 17 * y^2 = 532 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1191_119116


namespace NUMINAMATH_CALUDE_class_2003_ice_cream_picnic_student_ticket_cost_l1191_119176

/-- The cost of a student ticket for the Class of 2003 ice cream picnic -/
def student_ticket_cost : ℚ := sorry

/-- The theorem stating the cost of a student ticket for the Class of 2003 ice cream picnic -/
theorem class_2003_ice_cream_picnic_student_ticket_cost :
  let total_tickets : ℕ := 193
  let non_student_ticket_cost : ℚ := 3/2
  let total_revenue : ℚ := 413/2
  let student_tickets : ℕ := 83
  let non_student_tickets : ℕ := total_tickets - student_tickets
  student_ticket_cost * student_tickets + non_student_ticket_cost * non_student_tickets = total_revenue ∧
  student_ticket_cost = 1/2
  := by sorry

end NUMINAMATH_CALUDE_class_2003_ice_cream_picnic_student_ticket_cost_l1191_119176


namespace NUMINAMATH_CALUDE_correct_guesser_is_D_l1191_119179

-- Define the set of suspects
inductive Suspect : Type
| A | B | C | D | E | F

-- Define the set of passersby
inductive Passerby : Type
| A | B | C | D

-- Define a function to represent each passerby's guess
def guess (p : Passerby) (s : Suspect) : Prop :=
  match p with
  | Passerby.A => s = Suspect.D ∨ s = Suspect.E
  | Passerby.B => s ≠ Suspect.C
  | Passerby.C => s = Suspect.A ∨ s = Suspect.B ∨ s = Suspect.F
  | Passerby.D => s ≠ Suspect.D ∧ s ≠ Suspect.E ∧ s ≠ Suspect.F

-- Theorem statement
theorem correct_guesser_is_D :
  ∃! (thief : Suspect),
    ∃! (correct_passerby : Passerby),
      (∀ (p : Passerby), p ≠ correct_passerby → ¬guess p thief) ∧
      guess correct_passerby thief ∧
      correct_passerby = Passerby.D :=
by sorry

end NUMINAMATH_CALUDE_correct_guesser_is_D_l1191_119179


namespace NUMINAMATH_CALUDE_angle_with_touching_circles_theorem_l1191_119103

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an angle in 2D space --/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- Predicate to check if a circle touches a line internally --/
def touches_internally (c : Circle) (l : ℝ × ℝ → Prop) : Prop := sorry

/-- Predicate to check if two circles are non-intersecting --/
def non_intersecting (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on an angle --/
def on_angle (p : ℝ × ℝ) (a : Angle) : Prop := sorry

/-- Predicate to check if a point describes the arc of a circle --/
def describes_circle_arc (p : ℝ × ℝ) : Prop := sorry

theorem angle_with_touching_circles_theorem (a : Angle) (c1 c2 : Circle) 
  (h1 : touches_internally c1 a.side1)
  (h2 : touches_internally c2 a.side2)
  (h3 : non_intersecting c1 c2) :
  ∃ p : ℝ × ℝ, on_angle p a ∧ describes_circle_arc p := by
  sorry

end NUMINAMATH_CALUDE_angle_with_touching_circles_theorem_l1191_119103


namespace NUMINAMATH_CALUDE_distance_sum_equals_3root2_l1191_119117

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = -t ∧ y = 1 + t

-- Define point P in Cartesian coordinates
def point_P : ℝ × ℝ := (0, 1)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l t₁ A.1 A.2 ∧ circle_C A.1 A.2 ∧
    line_l t₂ B.1 B.2 ∧ circle_C B.1 B.2 ∧
    t₁ ≠ t₂

-- State the theorem
theorem distance_sum_equals_3root2 (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_equals_3root2_l1191_119117


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l1191_119111

theorem baseball_card_value_decrease : ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.6)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := (initial_value - value_after_second_year) / initial_value
  total_decrease = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l1191_119111


namespace NUMINAMATH_CALUDE_train_length_l1191_119142

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 225 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1191_119142


namespace NUMINAMATH_CALUDE_camera_imaging_formula_l1191_119163

/-- Given the camera imaging formula, prove the relationship between focal length,
    object distance, and image distance. -/
theorem camera_imaging_formula (f u v : ℝ) (hf : f ≠ 0) (hu : u ≠ 0) (hv : v ≠ 0) (hv_neq_f : v ≠ f) :
  1 / f = 1 / u + 1 / v → v = f * u / (u - f) := by
  sorry

end NUMINAMATH_CALUDE_camera_imaging_formula_l1191_119163


namespace NUMINAMATH_CALUDE_translation_right_3_units_l1191_119199

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateRight (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem translation_right_3_units :
  let A : Point := { x := 2, y := -1 }
  let A' : Point := translateRight A 3
  A'.x = 5 ∧ A'.y = -1 := by
sorry

end NUMINAMATH_CALUDE_translation_right_3_units_l1191_119199


namespace NUMINAMATH_CALUDE_square_area_relation_l1191_119122

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := a + b
  let area_I := (diagonal_I^2) / 2
  let area_II := 2 * area_I
  area_II = (a + b)^2 := by
sorry

end NUMINAMATH_CALUDE_square_area_relation_l1191_119122


namespace NUMINAMATH_CALUDE_total_oranges_equals_147_l1191_119187

/-- Represents the number of oranges picked by Mary on Monday -/
def mary_monday_oranges : ℕ := 14

/-- Represents the number of oranges picked by Jason on Monday -/
def jason_monday_oranges : ℕ := 41

/-- Represents the number of oranges picked by Amanda on Monday -/
def amanda_monday_oranges : ℕ := 56

/-- Represents the number of apples picked by Mary on Tuesday -/
def mary_tuesday_apples : ℕ := 22

/-- Represents the number of grapefruits picked by Jason on Tuesday -/
def jason_tuesday_grapefruits : ℕ := 15

/-- Represents the number of oranges picked by Amanda on Tuesday -/
def amanda_tuesday_oranges : ℕ := 36

/-- Represents the number of apples picked by Keith on Monday -/
def keith_monday_apples : ℕ := 38

/-- Represents the number of plums picked by Keith on Tuesday -/
def keith_tuesday_plums : ℕ := 47

/-- The total number of oranges picked over two days -/
def total_oranges : ℕ := mary_monday_oranges + jason_monday_oranges + amanda_monday_oranges + amanda_tuesday_oranges

theorem total_oranges_equals_147 : total_oranges = 147 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_equals_147_l1191_119187


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_three_l1191_119128

theorem greatest_common_divisor_of_three (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
    {d : ℕ | d ∣ 180 ∧ d ∣ n} = {d1, d2, d3}) →
  (Nat.gcd 180 n = 9) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_three_l1191_119128


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1191_119171

theorem necessary_not_sufficient_condition (x : ℝ) :
  (∀ y : ℝ, y > 2 → y > 1) ∧ (∃ z : ℝ, z > 1 ∧ z ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1191_119171


namespace NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l1191_119127

theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) :
  let circle_radius : ℝ := r
  let triangle_side_length : ℝ := r * Real.sqrt 3
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side_length ^ 2
  triangle_area = 75 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l1191_119127


namespace NUMINAMATH_CALUDE_point_c_coordinates_l1191_119193

/-- Given points A and B in ℝ², and a relationship between vectors AC and CB,
    prove that point C has specific coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (3, 0) → 
  C - A = -2 • (B - C) → 
  C = (4, -3) := by
sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l1191_119193


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1191_119114

theorem quadratic_equation_root (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - 65 = 0) ∧ (a * 5^2 + 3 * 5 - 65 = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1191_119114


namespace NUMINAMATH_CALUDE_descendants_characterization_l1191_119120

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The set of descendants of 1 -/
inductive Descendant : ℚ → Prop where
  | base : Descendant 1
  | left (x : ℚ) : Descendant x → Descendant (x + 1)
  | right (x : ℚ) : Descendant x → Descendant (x / (x + 1))

/-- Theorem: All descendants of 1 are of the form F_(n±1) / F_n, where n > 1 -/
theorem descendants_characterization (q : ℚ) :
  Descendant q ↔ ∃ n : ℕ, n > 1 ∧ (q = (fib (n + 1) : ℚ) / fib n ∨ q = (fib (n - 1) : ℚ) / fib n) :=
sorry

end NUMINAMATH_CALUDE_descendants_characterization_l1191_119120


namespace NUMINAMATH_CALUDE_F_2017_composition_l1191_119134

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

def F_comp (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => x
  | n+1 => F (F_comp n x)

theorem F_2017_composition (x : ℝ) : F_comp 2017 x = (x + 1)^(3^2017) - 1 := by
  sorry

end NUMINAMATH_CALUDE_F_2017_composition_l1191_119134


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1191_119158

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem unique_solution_factorial_equation :
  ∃! n : ℕ, 3 * n * factorial n + 2 * factorial n = 40320 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1191_119158


namespace NUMINAMATH_CALUDE_kittens_and_mice_count_l1191_119100

/-- The number of children carrying baskets -/
def num_children : ℕ := 12

/-- The number of baskets each child carries -/
def baskets_per_child : ℕ := 3

/-- The number of cats in each basket -/
def cats_per_basket : ℕ := 1

/-- The number of kittens each cat has -/
def kittens_per_cat : ℕ := 12

/-- The number of mice each kitten carries -/
def mice_per_kitten : ℕ := 4

/-- The total number of kittens and mice carried by the children -/
def total_kittens_and_mice : ℕ :=
  let total_baskets := num_children * baskets_per_child
  let total_cats := total_baskets * cats_per_basket
  let total_kittens := total_cats * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice

theorem kittens_and_mice_count : total_kittens_and_mice = 2160 := by
  sorry

end NUMINAMATH_CALUDE_kittens_and_mice_count_l1191_119100


namespace NUMINAMATH_CALUDE_prism_21_edges_9_faces_l1191_119192

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)

/-- Theorem: A prism with 21 edges has 9 faces. -/
theorem prism_21_edges_9_faces :
  ∀ p : Prism, p.edges = 21 → num_faces p = 9 := by
  sorry


end NUMINAMATH_CALUDE_prism_21_edges_9_faces_l1191_119192


namespace NUMINAMATH_CALUDE_calculate_total_earnings_l1191_119153

/-- Represents the number of days it takes for a person to complete the job alone. -/
structure WorkRate where
  days : ℕ
  days_pos : days > 0

/-- Represents the daily work rate of a person. -/
def daily_rate (w : WorkRate) : ℚ := 1 / w.days

/-- Calculates the total daily rate when multiple people work together. -/
def total_daily_rate (rates : List ℚ) : ℚ := rates.sum

/-- Represents the earnings of the workers. -/
structure Earnings where
  total : ℚ
  total_pos : total > 0

/-- Main theorem: Given the work rates and b's earnings, prove the total earnings. -/
theorem calculate_total_earnings
  (a b c : WorkRate)
  (h_a : a.days = 6)
  (h_b : b.days = 8)
  (h_c : c.days = 12)
  (b_earnings : ℚ)
  (h_b_earnings : b_earnings = 390)
  : ∃ (e : Earnings), e.total = 1170 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_earnings_l1191_119153


namespace NUMINAMATH_CALUDE_three_percentage_problem_l1191_119147

theorem three_percentage_problem (x y : ℝ) 
  (h1 : 3 = 0.25 * x) 
  (h2 : 3 = 0.50 * y) : 
  x - y = 6 ∧ x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_three_percentage_problem_l1191_119147


namespace NUMINAMATH_CALUDE_degree_of_3ab_l1191_119190

-- Define a monomial type
def Monomial := List (String × Nat)

-- Define a function to calculate the degree of a monomial
def degree (m : Monomial) : Nat :=
  m.foldl (fun acc (_, power) => acc + power) 0

-- Define our specific monomial 3ab
def monomial_3ab : Monomial := [("a", 1), ("b", 1)]

-- Theorem statement
theorem degree_of_3ab : degree monomial_3ab = 2 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_3ab_l1191_119190


namespace NUMINAMATH_CALUDE_most_reasonable_sampling_method_l1191_119196

/-- Represents the different sampling methods --/
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

/-- Represents the educational stages --/
inductive EducationalStage
| Primary
| JuniorHigh
| SeniorHigh

/-- Represents whether there are significant differences in vision conditions --/
def HasSignificantDifferences : Prop := True

/-- The most reasonable sampling method given the conditions --/
def MostReasonableSamplingMethod : SamplingMethod := SamplingMethod.StratifiedByEducationalStage

theorem most_reasonable_sampling_method
  (h1 : HasSignificantDifferences → ∀ (s1 s2 : EducationalStage), s1 ≠ s2 → ∃ (diff : ℝ), diff > 0)
  (h2 : ¬HasSignificantDifferences → ∀ (gender1 gender2 : Bool), ∀ (ε : ℝ), ε > 0 → ∃ (diff : ℝ), diff < ε)
  : MostReasonableSamplingMethod = SamplingMethod.StratifiedByEducationalStage :=
by sorry

end NUMINAMATH_CALUDE_most_reasonable_sampling_method_l1191_119196


namespace NUMINAMATH_CALUDE_max_value_at_e_l1191_119121

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_at_e :
  ∀ x : ℝ, x > 0 → f x ≤ f (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_e_l1191_119121


namespace NUMINAMATH_CALUDE_min_grass_seed_amount_is_75_l1191_119172

/-- Represents a bag of grass seed with its weight and price -/
structure GrassSeedBag where
  weight : ℕ
  price : ℚ

/-- Finds the minimum amount of grass seed that can be purchased given the constraints -/
def minGrassSeedAmount (bags : List GrassSeedBag) (maxWeight : ℕ) (exactCost : ℚ) : ℕ :=
  sorry

theorem min_grass_seed_amount_is_75 :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 13.82 },
    { weight := 10, price := 20.43 },
    { weight := 25, price := 32.25 }
  ]
  let maxWeight : ℕ := 80
  let exactCost : ℚ := 98.75

  minGrassSeedAmount bags maxWeight exactCost = 75 := by sorry

end NUMINAMATH_CALUDE_min_grass_seed_amount_is_75_l1191_119172


namespace NUMINAMATH_CALUDE_circles_intersect_l1191_119112

theorem circles_intersect (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 5) 
  (h₂ : r₂ = 3) 
  (h₃ : d = 7) : 
  (r₁ - r₂ < d) ∧ (d < r₁ + r₂) := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_circles_intersect_l1191_119112


namespace NUMINAMATH_CALUDE_sports_club_purchase_l1191_119181

/-- The price difference between a basketball and a soccer ball -/
def price_difference : ℕ := 30

/-- The budget for soccer balls -/
def soccer_budget : ℕ := 1500

/-- The budget for basketballs -/
def basketball_budget : ℕ := 2400

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 100

/-- The minimum discount on basketballs -/
def min_discount : ℕ := 25

/-- The maximum discount on basketballs -/
def max_discount : ℕ := 35

/-- The price of a soccer ball -/
def soccer_price : ℕ := 50

/-- The price of a basketball -/
def basketball_price : ℕ := 80

theorem sports_club_purchase :
  ∀ (m : ℕ), min_discount ≤ m → m ≤ max_discount →
  (∃ (y : ℕ), y ≤ total_balls ∧ 3 * (total_balls - y) ≤ y ∧
    (∀ (z : ℕ), z ≤ total_balls → 3 * (total_balls - z) ≤ z →
      (if 30 < m then
        (basketball_price - m) * y + soccer_price * (total_balls - y) ≤ (basketball_price - m) * z + soccer_price * (total_balls - z)
      else if m < 30 then
        (basketball_price - m) * y + soccer_price * (total_balls - y) ≤ (basketball_price - m) * z + soccer_price * (total_balls - z)
      else
        (basketball_price - m) * y + soccer_price * (total_balls - y) = (basketball_price - m) * z + soccer_price * (total_balls - z)))) ∧
  basketball_price = soccer_price + price_difference ∧
  soccer_budget / soccer_price = basketball_budget / basketball_price := by
  sorry

end NUMINAMATH_CALUDE_sports_club_purchase_l1191_119181


namespace NUMINAMATH_CALUDE_f_derivative_l1191_119191

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def f (x : ℝ) : ℝ :=
  binomial 4 0 - binomial 4 1 * x + binomial 4 2 * x^2 - binomial 4 3 * x^3 + binomial 4 4 * x^4

theorem f_derivative (x : ℝ) : deriv f x = 4 * (-1 + x)^3 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l1191_119191


namespace NUMINAMATH_CALUDE_geometric_mean_scaling_l1191_119195

theorem geometric_mean_scaling (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) 
  (h₅ : a₅ > 0) (h₆ : a₆ > 0) (h₇ : a₇ > 0) (h₈ : a₈ > 0) :
  (((5 * a₁) * (5 * a₂) * (5 * a₃) * (5 * a₄) * (5 * a₅) * (5 * a₆) * (5 * a₇) * (5 * a₈)) ^ (1/8 : ℝ)) = 
  5 * ((a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇ * a₈) ^ (1/8 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_scaling_l1191_119195


namespace NUMINAMATH_CALUDE_highlighter_spent_theorem_l1191_119157

def total_money : ℝ := 150
def sharpener_price : ℝ := 3
def notebook_price : ℝ := 7
def eraser_price : ℝ := 2
def sharpener_count : ℕ := 5
def notebook_count : ℕ := 6
def eraser_count : ℕ := 15

def heaven_spent : ℝ := sharpener_price * sharpener_count + notebook_price * notebook_count

def money_left_after_heaven : ℝ := total_money - heaven_spent

def brother_eraser_spent : ℝ := eraser_price * eraser_count

theorem highlighter_spent_theorem :
  money_left_after_heaven - brother_eraser_spent = 63 := by sorry

end NUMINAMATH_CALUDE_highlighter_spent_theorem_l1191_119157


namespace NUMINAMATH_CALUDE_triangle_arctan_sum_l1191_119183

theorem triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let angle_C := 2 * Real.pi / 3
  (a^2 + b^2 + c^2 = 2 * a * b * Real.cos angle_C + 2 * b * c + 2 * c * a) →
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_arctan_sum_l1191_119183


namespace NUMINAMATH_CALUDE_g_of_one_eq_neg_two_l1191_119108

theorem g_of_one_eq_neg_two :
  let g : ℝ → ℝ := fun x ↦ x^3 - x^2 - 2*x
  g 1 = -2 := by sorry

end NUMINAMATH_CALUDE_g_of_one_eq_neg_two_l1191_119108


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l1191_119177

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_valid_votes : ℕ)
  (h_total : total_votes = 560000)
  (h_percentage : candidate_a_percentage = 70 / 100)
  (h_valid_votes : candidate_a_valid_votes = 333200) :
  (total_votes - (candidate_a_valid_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l1191_119177


namespace NUMINAMATH_CALUDE_x_equals_four_l1191_119133

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  fun (a, b) (c, d) => (a - c, b + d)

/-- Theorem stating that x = 4 given the conditions -/
theorem x_equals_four :
  ∀ y : ℤ, star (4, 1) (1, -2) = star (x, y) (1, 4) → x = 4 :=
by
  sorry

#check x_equals_four

end NUMINAMATH_CALUDE_x_equals_four_l1191_119133


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l1191_119186

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2^(sequence_a n) + 2

theorem divisibility_in_sequence (m n : ℕ) (h : m < n) :
  ∃ k : ℕ, sequence_a n = k * sequence_a m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l1191_119186


namespace NUMINAMATH_CALUDE_dancing_preference_fraction_l1191_119148

def total_students : ℕ := 200
def like_dancing_percent : ℚ := 70 / 100
def dislike_dancing_percent : ℚ := 30 / 100
def honest_like_percent : ℚ := 85 / 100
def dishonest_like_percent : ℚ := 15 / 100
def honest_dislike_percent : ℚ := 80 / 100
def dishonest_dislike_percent : ℚ := 20 / 100

theorem dancing_preference_fraction :
  let like_dancing := (like_dancing_percent * total_students : ℚ)
  let dislike_dancing := (dislike_dancing_percent * total_students : ℚ)
  let say_like := (honest_like_percent * like_dancing + dishonest_dislike_percent * dislike_dancing : ℚ)
  let actually_dislike_say_like := (dishonest_dislike_percent * dislike_dancing : ℚ)
  actually_dislike_say_like / say_like = 12 / 131 := by
  sorry

end NUMINAMATH_CALUDE_dancing_preference_fraction_l1191_119148


namespace NUMINAMATH_CALUDE_complex_multiplication_l1191_119169

theorem complex_multiplication : ∃ (i : ℂ), i^2 = -1 ∧ (3 - 4*i) * (-7 + 2*i) = -13 + 34*i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1191_119169


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1191_119109

theorem unique_solution_condition (a : ℝ) : 
  (∃! x, 0 ≤ x^2 + a*x + 6 ∧ x^2 + a*x + 6 ≤ 4) ↔ (a = 2*Real.sqrt 2 ∨ a = -2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1191_119109


namespace NUMINAMATH_CALUDE_cyclic_sum_nonnegative_l1191_119138

theorem cyclic_sum_nonnegative (a b c k : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hk_lower : k ≥ 0) (hk_upper : k < 2) : 
  (a^2 - b*c)/(b^2 + c^2 + k*a^2) + 
  (b^2 - a*c)/(a^2 + c^2 + k*b^2) + 
  (c^2 - a*b)/(a^2 + b^2 + k*c^2) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_nonnegative_l1191_119138


namespace NUMINAMATH_CALUDE_field_trip_students_field_trip_problem_l1191_119197

theorem field_trip_students (van_capacity : ℕ) (num_vans : ℕ) (num_adults : ℕ) : ℕ :=
  let total_capacity := van_capacity * num_vans
  let num_students := total_capacity - num_adults
  num_students

theorem field_trip_problem : 
  field_trip_students 4 2 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_field_trip_problem_l1191_119197


namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l1191_119119

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    ∀ k : ℝ, (b^2) / (a^2 + c^2) ≤ k → k ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l1191_119119


namespace NUMINAMATH_CALUDE_diamond_equation_l1191_119149

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_equation : 
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_l1191_119149


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_eighteen_l1191_119178

theorem arithmetic_expression_equals_eighteen :
  8 / 2 - 3 - 10 + 3 * 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_eighteen_l1191_119178


namespace NUMINAMATH_CALUDE_amount_saved_is_30_l1191_119152

/-- Calculates the amount saved after clearing debt given income and expenses -/
def amountSavedAfterDebt (monthlyIncome : ℕ) (initialExpense : ℕ) (reducedExpense : ℕ) : ℕ :=
  let initialPeriod := 6
  let reducedPeriod := 4
  let initialDebt := initialPeriod * initialExpense - initialPeriod * monthlyIncome
  let totalIncome := (initialPeriod + reducedPeriod) * monthlyIncome
  let totalExpense := initialPeriod * initialExpense + reducedPeriod * reducedExpense
  totalIncome - (totalExpense + initialDebt)

/-- Theorem: Given the specified income and expenses, the amount saved after clearing debt is 30 -/
theorem amount_saved_is_30 :
  amountSavedAfterDebt 69 70 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_amount_saved_is_30_l1191_119152


namespace NUMINAMATH_CALUDE_gcd_50404_40303_l1191_119131

theorem gcd_50404_40303 : Nat.gcd 50404 40303 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50404_40303_l1191_119131


namespace NUMINAMATH_CALUDE_simplify_expression_l1191_119165

theorem simplify_expression : (5 * 10^9) / (2 * 10^5 * 5) = 5000 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1191_119165


namespace NUMINAMATH_CALUDE_ravi_coins_value_is_350_l1191_119188

/-- Calculates the total value of Ravi's coins given the number of nickels --/
def raviCoinsValue (nickels : ℕ) : ℚ :=
  let quarters := nickels + 2
  let dimes := quarters + 4
  let nickelValue : ℚ := 5 / 100
  let quarterValue : ℚ := 25 / 100
  let dimeValue : ℚ := 10 / 100
  nickels * nickelValue + quarters * quarterValue + dimes * dimeValue

/-- Theorem stating that Ravi's coins are worth $3.50 given the conditions --/
theorem ravi_coins_value_is_350 : raviCoinsValue 6 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ravi_coins_value_is_350_l1191_119188


namespace NUMINAMATH_CALUDE_unique_value_in_set_l1191_119146

theorem unique_value_in_set (a : ℝ) : 1 ∈ ({a, a + 1, a^2} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_in_set_l1191_119146


namespace NUMINAMATH_CALUDE_yoghurt_cost_l1191_119173

/-- Given Tara's purchase of ice cream and yoghurt, prove the cost of each yoghurt carton. -/
theorem yoghurt_cost (ice_cream_cartons : ℕ) (yoghurt_cartons : ℕ) 
  (ice_cream_cost : ℕ) (price_difference : ℕ) :
  ice_cream_cartons = 19 →
  yoghurt_cartons = 4 →
  ice_cream_cost = 7 →
  price_difference = 129 →
  (ice_cream_cartons * ice_cream_cost - price_difference) / yoghurt_cartons = 1 := by
  sorry

end NUMINAMATH_CALUDE_yoghurt_cost_l1191_119173


namespace NUMINAMATH_CALUDE_perpendicular_parallel_relationships_l1191_119115

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_relationships 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β) :
  (parallel α β → line_perpendicular l m) ∧
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_relationships_l1191_119115


namespace NUMINAMATH_CALUDE_cyclist_club_members_count_l1191_119182

/-- The set of digits that can be used in the identification numbers. -/
def ValidDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 9}

/-- The number of digits in each identification number. -/
def IdentificationNumberLength : Nat := 3

/-- The total number of possible identification numbers. -/
def TotalIdentificationNumbers : Nat := ValidDigits.card ^ IdentificationNumberLength

/-- Theorem stating that the total number of possible identification numbers is 512. -/
theorem cyclist_club_members_count :
  TotalIdentificationNumbers = 512 := by sorry

end NUMINAMATH_CALUDE_cyclist_club_members_count_l1191_119182


namespace NUMINAMATH_CALUDE_original_true_implies_contrapositive_true_l1191_119155

-- Define a proposition type
variable (P Q : Prop)

-- Define the contrapositive of an implication
def contrapositive (P Q : Prop) : Prop := ¬Q → ¬P

-- Theorem: If the original proposition is true, then its contrapositive is also true
theorem original_true_implies_contrapositive_true (h : P → Q) : contrapositive P Q :=
  sorry

end NUMINAMATH_CALUDE_original_true_implies_contrapositive_true_l1191_119155


namespace NUMINAMATH_CALUDE_least_power_congruence_l1191_119110

theorem least_power_congruence (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 195 → 3^m % 143^2 ≠ 1) ∧ 
  3^195 % 143^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_power_congruence_l1191_119110


namespace NUMINAMATH_CALUDE_inequality_properties_l1191_119136

theorem inequality_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  (∀ c, a + c > b + c) ∧
  (a^2 > b^2) ∧
  (Real.sqrt a > Real.sqrt b) ∧
  (∃ c, a * c ≤ b * c) := by
sorry

end NUMINAMATH_CALUDE_inequality_properties_l1191_119136


namespace NUMINAMATH_CALUDE_money_redistribution_l1191_119166

-- Define the initial amounts for each person
variable (a b c d : ℝ)

-- Define the redistribution function
def redistribute (x y z w : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (x - (y + z + w), 2*y, 2*z, 2*w)

-- Theorem statement
theorem money_redistribution (h1 : c = 24) :
  let (a', b', c', d') := redistribute a b c d
  let (a'', b'', c'', d'') := redistribute b' a' c' d'
  let (a''', b''', c''', d''') := redistribute c'' a'' b'' d''
  let (a_final, b_final, c_final, d_final) := redistribute d''' a''' b''' c'''
  c_final = c → a + b + c + d = 96 := by
  sorry

end NUMINAMATH_CALUDE_money_redistribution_l1191_119166


namespace NUMINAMATH_CALUDE_equation_solution_l1191_119161

-- Define the operation "*"
def star_op (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- State the theorem
theorem equation_solution :
  ∃! x : ℝ, star_op (x - 4) 1 = 0 ∧ x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1191_119161


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1191_119113

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
    (∀ x, 3 * x^2 = 5 * x - 1) →
    (∀ x, a * x^2 + b * x + c = 0) →
    (∀ x, 3 * x^2 - 5 * x + 1 = 0) →
    a = 3 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1191_119113


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1191_119170

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1 / x) = 1.5) :
  x^2 + (1 / x^2) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1191_119170


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1191_119123

/-- Given a sum P put at simple interest rate R% for 1 year, 
    if increasing the rate by 6% results in Rs. 30 more interest, 
    then P = 500. -/
theorem simple_interest_problem (P R : ℝ) 
  (h1 : P * (R + 6) / 100 = P * R / 100 + 30) : 
  P = 500 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1191_119123


namespace NUMINAMATH_CALUDE_alternating_ones_zeros_composite_l1191_119126

/-- The number formed by k+1 ones with k zeros interspersed between them -/
def alternating_ones_zeros (k : ℕ) : ℕ :=
  (10^(k+1) - 1) / 9

/-- Theorem stating that the alternating_ones_zeros number is composite for k ≥ 2 -/
theorem alternating_ones_zeros_composite (k : ℕ) (h : k ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ alternating_ones_zeros k = a * b :=
sorry


end NUMINAMATH_CALUDE_alternating_ones_zeros_composite_l1191_119126


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1191_119139

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1191_119139


namespace NUMINAMATH_CALUDE_hexagon_circle_comparison_l1191_119125

theorem hexagon_circle_comparison : ∃ (h r : ℝ),
  h > 0 ∧ r > 0 ∧
  (3 * Real.sqrt 3 / 2) * h^2 = 6 * h ∧  -- Hexagon area equals perimeter
  π * r^2 = 2 * π * r ∧                  -- Circle area equals perimeter
  (Real.sqrt 3 / 2) * h = r ∧            -- Apothem equals radius
  r = 2 := by sorry

end NUMINAMATH_CALUDE_hexagon_circle_comparison_l1191_119125


namespace NUMINAMATH_CALUDE_gift_box_weight_l1191_119175

/-- The weight of an empty gift box -/
def empty_box_weight (num_tangerines : ℕ) (tangerine_weight : ℝ) (total_weight : ℝ) : ℝ :=
  total_weight - (num_tangerines : ℝ) * tangerine_weight

/-- Theorem: The weight of the empty gift box is 0.46 kg -/
theorem gift_box_weight :
  empty_box_weight 30 0.36 11.26 = 0.46 := by sorry

end NUMINAMATH_CALUDE_gift_box_weight_l1191_119175


namespace NUMINAMATH_CALUDE_mode_of_interest_groups_l1191_119135

def interest_groups : List Nat := [4, 7, 5, 4, 6, 4, 5]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_interest_groups :
  mode interest_groups = 4 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_interest_groups_l1191_119135


namespace NUMINAMATH_CALUDE_quadratic_domain_range_existence_l1191_119140

/-- 
Given a quadratic function f(x) = -1/2 * x^2 + x + a, where a is a constant,
there exist real numbers m and n (with m < n) such that the domain of f is [m, n]
and the range is [3m, 3n] if and only if -2 < a ≤ 5/2.
-/
theorem quadratic_domain_range_existence (a : ℝ) :
  (∃ (m n : ℝ), m < n ∧
    (∀ x, x ∈ Set.Icc m n ↔ -1/2 * x^2 + x + a ∈ Set.Icc (3*m) (3*n)) ∧
    (∀ y, y ∈ Set.Icc (3*m) (3*n) → ∃ x ∈ Set.Icc m n, y = -1/2 * x^2 + x + a)) ↔
  -2 < a ∧ a ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_domain_range_existence_l1191_119140


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1191_119180

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1191_119180


namespace NUMINAMATH_CALUDE_intersecting_line_equation_l1191_119141

/-- A line that intersects a circle and a hyperbola with specific properties -/
structure IntersectingLine (a : ℝ) where
  m : ℝ
  b : ℝ
  intersects_circle : ∀ x y, y = m * x + b → x^2 + y^2 = a^2
  intersects_hyperbola : ∀ x y, y = m * x + b → x^2 - y^2 = a^2
  trisects : ∀ (x₁ x₂ x₃ x₄ : ℝ),
    (x₁^2 + (m * x₁ + b)^2 = a^2) →
    (x₂^2 + (m * x₂ + b)^2 = a^2) →
    (x₃^2 - (m * x₃ + b)^2 = a^2) →
    (x₄^2 - (m * x₄ + b)^2 = a^2) →
    (x₁ - x₂)^2 = (1/9) * (x₃ - x₄)^2

/-- The equation of the intersecting line is y = ±(2√5/5)x or y = ±(2√5/5)a -/
theorem intersecting_line_equation (a : ℝ) (l : IntersectingLine a) :
  (l.m = 2 * Real.sqrt 5 / 5 ∧ l.b = 0) ∨
  (l.m = -2 * Real.sqrt 5 / 5 ∧ l.b = 0) ∨
  (l.m = 0 ∧ l.b = 2 * a * Real.sqrt 5 / 5) ∨
  (l.m = 0 ∧ l.b = -2 * a * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_equation_l1191_119141


namespace NUMINAMATH_CALUDE_tournament_results_count_l1191_119150

-- Define the teams
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

-- Define a match result
inductive MatchResult : Type
| Win : Team → MatchResult
| Loss : Team → MatchResult

-- Define a tournament result
structure TournamentResult : Type :=
(saturday1 : MatchResult)  -- E vs F
(saturday2 : MatchResult)  -- G vs H
(sunday1 : MatchResult)    -- 1st vs 2nd
(sunday2 : MatchResult)    -- 3rd vs 4th

def count_tournament_results : ℕ :=
  -- The actual count will be implemented in the proof
  sorry

theorem tournament_results_count :
  count_tournament_results = 16 := by
  sorry

end NUMINAMATH_CALUDE_tournament_results_count_l1191_119150


namespace NUMINAMATH_CALUDE_fred_seashells_l1191_119101

theorem fred_seashells (initial_seashells : ℕ) (given_seashells : ℕ) : 
  initial_seashells = 47 → given_seashells = 25 → initial_seashells - given_seashells = 22 := by
  sorry

end NUMINAMATH_CALUDE_fred_seashells_l1191_119101


namespace NUMINAMATH_CALUDE_junior_freshman_ratio_l1191_119185

theorem junior_freshman_ratio (f j : ℕ) (hf : f > 0) (hj : j > 0)
  (h_participants : (1 : ℚ) / 4 * f = (1 : ℚ) / 2 * j) :
  j / f = (1 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_junior_freshman_ratio_l1191_119185


namespace NUMINAMATH_CALUDE_CD_parallel_BE_l1191_119164

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define points C and D
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (2, 0)

-- Define a line passing through C and a point (x, y) on Γ
def line_through_C (x y : ℝ) : Set (ℝ × ℝ) :=
  {(t, u) | ∃ (k : ℝ), u - C.2 = k * (t - C.1) ∧ t ≠ C.1}

-- Define point A as an intersection of the line and Γ
def A (x y : ℝ) : ℝ × ℝ :=
  (x, y)

-- Define point B as the other intersection of the line and Γ
def B (x y : ℝ) : ℝ × ℝ :=
  sorry

-- Define point E as the intersection of AD and x=3
def E (x y : ℝ) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem CD_parallel_BE (x y : ℝ) :
  Γ x y →
  (x, y) ∈ line_through_C x y →
  (B x y).1 ≠ x →
  (let slope_CD := (D.2 - C.2) / (D.1 - C.1)
   let slope_BE := (E x y).2 / ((E x y).1 - (B x y).1)
   slope_CD = slope_BE) :=
sorry

end NUMINAMATH_CALUDE_CD_parallel_BE_l1191_119164


namespace NUMINAMATH_CALUDE_integer_cube_between_zero_and_nine_l1191_119156

theorem integer_cube_between_zero_and_nine (a : ℤ) : 0 < a^3 ∧ a^3 < 9 → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_cube_between_zero_and_nine_l1191_119156


namespace NUMINAMATH_CALUDE_fraction_simplification_l1191_119184

theorem fraction_simplification (x : ℝ) (h : x ≠ 0) :
  ((x + 3)^2 + (x + 3)*(x - 3)) / (2*x) = x + 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1191_119184


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_odds_l1191_119160

theorem largest_of_four_consecutive_odds (x : ℤ) : 
  (x % 2 = 1) →                           -- x is odd
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →  -- average is 24
  (x + 6 = 27) :=                         -- largest number is 27
by sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_odds_l1191_119160


namespace NUMINAMATH_CALUDE_min_a_is_neg_two_l1191_119154

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- The theorem stating that -2 is the minimum value of a for which f(x) is always non-negative -/
theorem min_a_is_neg_two :
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) → a ≥ -2) ∧
  (∀ x : ℝ, f (-2) x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_min_a_is_neg_two_l1191_119154


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1191_119143

/-- Given that the solution set of ax^2 + bx + 4 > 0 is (-1, 2),
    prove that the solution set of ax + b + 4 > 0 is (-∞, 3) -/
theorem solution_set_inequality (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + 4 > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, a*x + b + 4 > 0 ↔ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1191_119143


namespace NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l1191_119198

theorem sum_positive_implies_one_positive (a b : ℝ) : a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l1191_119198


namespace NUMINAMATH_CALUDE_integral_proof_l1191_119118

theorem integral_proof (x C : ℝ) : 
  (deriv (λ x => 1 / (2 * (2 * Real.sin x - 3 * Real.cos x)^2) + C)) x = 
  (2 * Real.cos x + 3 * Real.sin x) / (2 * Real.sin x - 3 * Real.cos x)^3 := by
  sorry

end NUMINAMATH_CALUDE_integral_proof_l1191_119118


namespace NUMINAMATH_CALUDE_system_integer_solutions_l1191_119124

theorem system_integer_solutions (a b c d : ℤ) 
  (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) : 
  (a * d - b * c = 1) ∨ (a * d - b * c = -1) := by sorry

end NUMINAMATH_CALUDE_system_integer_solutions_l1191_119124


namespace NUMINAMATH_CALUDE_quotient_derivative_property_l1191_119107

/-- Two differentiable functions satisfying the property that the derivative of their quotient
    is equal to the quotient of their derivatives. -/
theorem quotient_derivative_property (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.exp (4 * x)
  let g : ℝ → ℝ := fun x => Real.exp (2 * x)
  (deriv (f / g)) x = (deriv f x) / (deriv g x) := by sorry

end NUMINAMATH_CALUDE_quotient_derivative_property_l1191_119107


namespace NUMINAMATH_CALUDE_f_range_l1191_119130

def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem f_range : ∀ x ∈ Set.Icc 0 3, 1 ≤ f x ∧ f x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_f_range_l1191_119130


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l1191_119137

/-- Prove that vectors a, b, and c are coplanar -/
theorem vectors_are_coplanar :
  let a : ℝ × ℝ × ℝ := (1, 2, -3)
  let b : ℝ × ℝ × ℝ := (-2, -4, 6)
  let c : ℝ × ℝ × ℝ := (1, 0, 5)
  ∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_vectors_are_coplanar_l1191_119137


namespace NUMINAMATH_CALUDE_valid_parameterizations_l1191_119189

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 4

-- Define the parameterizations
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)
def param_C (t : ℝ) : ℝ × ℝ := (-1 + 2 * t, -6 + 4 * t)
def param_D (t : ℝ) : ℝ × ℝ := (3 + t, 2 + 3 * t)
def param_E (t : ℝ) : ℝ × ℝ := (-4 - 2 * t, -12 - 4 * t)

-- Theorem stating which parameterizations are valid
theorem valid_parameterizations :
  (∀ t, line_equation (param_A t).1 (param_A t).2) ∧
  (∀ t, line_equation (param_B t).1 (param_B t).2) ∧
  ¬(∀ t, line_equation (param_C t).1 (param_C t).2) ∧
  ¬(∀ t, line_equation (param_D t).1 (param_D t).2) ∧
  ¬(∀ t, line_equation (param_E t).1 (param_E t).2) := by
  sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l1191_119189


namespace NUMINAMATH_CALUDE_john_streaming_hours_l1191_119102

/-- Calculates the number of hours streamed per day given the total weekly earnings,
    hourly rate, and number of streaming days per week. -/
def hours_streamed_per_day (weekly_earnings : ℕ) (hourly_rate : ℕ) (streaming_days : ℕ) : ℚ :=
  (weekly_earnings : ℚ) / (hourly_rate : ℚ) / (streaming_days : ℚ)

/-- Proves that John streams 4 hours per day given the problem conditions. -/
theorem john_streaming_hours :
  let weekly_earnings := 160
  let hourly_rate := 10
  let days_per_week := 7
  let days_off := 3
  let streaming_days := days_per_week - days_off
  hours_streamed_per_day weekly_earnings hourly_rate streaming_days = 4 := by
  sorry

#eval hours_streamed_per_day 160 10 4

end NUMINAMATH_CALUDE_john_streaming_hours_l1191_119102


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l1191_119145

-- Define a fair coin
def fair_coin_prob : ℚ := 1 / 2

-- Define the number of flips
def total_flips : ℕ := 5

-- Define the number of heads for the first probability
def heads_count_1 : ℕ := 3

-- Define the number of heads for the second probability
def heads_count_2 : ℕ := 5

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Function to calculate probability of exactly k heads in n flips
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem coin_flip_probability_difference : 
  (prob_k_heads total_flips heads_count_1 fair_coin_prob - 
   prob_k_heads total_flips heads_count_2 fair_coin_prob) = 9 / 32 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l1191_119145


namespace NUMINAMATH_CALUDE_comic_book_problem_l1191_119129

theorem comic_book_problem (x y : ℕ) : 
  (y + 7 = 5 * (x - 7)) →
  (y - 9 = 3 * (x + 9)) →
  (x = 39 ∧ y = 153) := by
sorry

end NUMINAMATH_CALUDE_comic_book_problem_l1191_119129


namespace NUMINAMATH_CALUDE_adam_savings_l1191_119151

theorem adam_savings (x : ℝ) : x + 13 = 92 → x = 79 := by
  sorry

end NUMINAMATH_CALUDE_adam_savings_l1191_119151


namespace NUMINAMATH_CALUDE_prob_heart_diamond_standard_deck_l1191_119105

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (num_red_suits : ℕ)
  (num_black_suits : ℕ)

/-- Standard deck properties -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    num_red_suits := 2,
    num_black_suits := 2 }

/-- Probability of drawing a heart first and a diamond second -/
def prob_heart_then_diamond (d : Deck) : ℚ :=
  (d.cards_per_suit : ℚ) / (d.total_cards : ℚ) *
  (d.cards_per_suit : ℚ) / ((d.total_cards - 1) : ℚ)

/-- Theorem stating the probability of drawing a heart then a diamond -/
theorem prob_heart_diamond_standard_deck :
  prob_heart_then_diamond standard_deck = 169 / 2652 :=
sorry

end NUMINAMATH_CALUDE_prob_heart_diamond_standard_deck_l1191_119105


namespace NUMINAMATH_CALUDE_al_original_portion_l1191_119194

/-- Represents the investment scenario with four participants --/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ
  dave : ℝ

/-- The investment scenario satisfies the given conditions --/
def ValidInvestment (i : Investment) : Prop :=
  i.al + i.betty + i.clare + i.dave = 1200 ∧
  (i.al - 150) + (2 * i.betty) + (2 * i.clare) + (3 * i.dave) = 1800

/-- Theorem stating that Al's original portion was $450 --/
theorem al_original_portion (i : Investment) (h : ValidInvestment i) : i.al = 450 := by
  sorry

#check al_original_portion

end NUMINAMATH_CALUDE_al_original_portion_l1191_119194


namespace NUMINAMATH_CALUDE_sphere_surface_area_containing_cuboid_l1191_119167

theorem sphere_surface_area_containing_cuboid (a b c : ℝ) (S : ℝ) :
  a = 3 → b = 4 → c = 5 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_containing_cuboid_l1191_119167


namespace NUMINAMATH_CALUDE_a_formula_l1191_119106

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 2
  | (n + 2) => 2 * a (n + 1) - a n + 2

theorem a_formula (n : ℕ) : a n = n^2 + n := by
  sorry

end NUMINAMATH_CALUDE_a_formula_l1191_119106


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_three_factors_l1191_119159

/-- Two natural numbers are consecutive primes if they are both prime and there is no prime between them. -/
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ k, p < k → k < q → ¬Nat.Prime k

/-- A natural number is the product of at least three factors greater than 1 if it can be written as the product of three or more natural numbers, each greater than 1. -/
def ProductOfAtLeastThreeFactors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ n = a * b * c

/-- For any two consecutive odd prime numbers, their sum is a product of at least three positive integers greater than 1. -/
theorem sum_of_consecutive_odd_primes_has_three_factors (p q : ℕ) :
  ConsecutivePrimes p q → Odd p → Odd q → ProductOfAtLeastThreeFactors (p + q) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_primes_has_three_factors_l1191_119159


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1191_119132

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 * x * 2*x = 64) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1191_119132
