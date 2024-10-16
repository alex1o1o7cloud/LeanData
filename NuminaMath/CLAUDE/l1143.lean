import Mathlib

namespace NUMINAMATH_CALUDE_complete_square_sum_l1143_114313

theorem complete_square_sum (a b c : ℝ) (r s : ℝ) :
  (6 * a^2 - 30 * a - 36 = 0) →
  ((a + r)^2 = s) →
  (6 * a^2 - 30 * a - 36 = 6 * ((a + r)^2 - s)) →
  (r + s = 9.75) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1143_114313


namespace NUMINAMATH_CALUDE_distance_between_points_l1143_114360

theorem distance_between_points : 
  let x₁ : ℝ := 6
  let y₁ : ℝ := -18
  let x₂ : ℝ := 3
  let y₂ : ℝ := 9
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 738 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1143_114360


namespace NUMINAMATH_CALUDE_jim_journey_l1143_114356

theorem jim_journey (total_journey : ℕ) (remaining_miles : ℕ) 
  (h1 : total_journey = 1200)
  (h2 : remaining_miles = 558) :
  total_journey - remaining_miles = 642 := by
sorry

end NUMINAMATH_CALUDE_jim_journey_l1143_114356


namespace NUMINAMATH_CALUDE_businessmen_drinks_l1143_114393

theorem businessmen_drinks (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  tea = 14 →
  both = 7 →
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_drinks_l1143_114393


namespace NUMINAMATH_CALUDE_dans_remaining_limes_l1143_114310

/-- Given that Dan initially had 9 limes and gave away 4 limes, prove that he now has 5 limes. -/
theorem dans_remaining_limes (initial_limes : ℕ) (given_away : ℕ) (h1 : initial_limes = 9) (h2 : given_away = 4) :
  initial_limes - given_away = 5 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_limes_l1143_114310


namespace NUMINAMATH_CALUDE_problem_solution_l1143_114363

def A (a : ℝ) := { x : ℝ | a - 1 ≤ x ∧ x ≤ a + 1 }
def B := { x : ℝ | -1 ≤ x ∧ x ≤ 4 }

theorem problem_solution :
  (∀ a : ℝ, a = 2 → A a ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 4 }) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) → 0 ≤ a ∧ a ≤ 3) ∧
  (∀ a : ℝ, A a ∪ B = B → 0 ≤ a ∧ a ≤ 3) ∧
  (∀ a : ℝ, A a ∩ B = ∅ → a < -2 ∨ a > 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1143_114363


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1143_114383

/-- The circle C in the xy-plane -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

/-- Point A -/
def A : ℝ × ℝ := (-1, 4)

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nontrivial : a ≠ 0 ∨ b ≠ 0

/-- A line is tangent to the circle C -/
def isTangent (l : Line) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ C ∧ l.a * p.1 + l.b * p.2 + l.c = 0 ∧
    ∀ q : ℝ × ℝ, q ∈ C → q ≠ p → l.a * q.1 + l.b * q.2 + l.c ≠ 0

/-- The tangent line passes through point A -/
def passesThroughA (l : Line) : Prop :=
  l.a * A.1 + l.b * A.2 + l.c = 0

theorem tangent_line_equation :
  ∀ l : Line, isTangent l → passesThroughA l →
    (l.a = 0 ∧ l.b = 1 ∧ l.c = -4) ∨ (l.a = 3 ∧ l.b = 4 ∧ l.c = -13) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1143_114383


namespace NUMINAMATH_CALUDE_b_work_time_l1143_114378

-- Define the work rates for A, B, C, and D
def A : ℚ := 1 / 5
def C : ℚ := 2 / 5 - A
def B : ℚ := 1 / 4 - C
def D : ℚ := 1 / 2 - B - C

-- State the theorem
theorem b_work_time : (1 : ℚ) / B = 20 := by sorry

end NUMINAMATH_CALUDE_b_work_time_l1143_114378


namespace NUMINAMATH_CALUDE_min_distance_to_point_l1143_114303

/-- The line equation ax + by + 1 = 0 -/
def line_equation (a b x y : ℝ) : Prop := a * x + b * y + 1 = 0

/-- The circle equation x^2 + y^2 + 4x + 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

/-- The line always bisects the circumference of the circle -/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a b x y → circle_equation x y

/-- The theorem to be proved -/
theorem min_distance_to_point (a b : ℝ) 
  (h : line_bisects_circle a b) : 
  (∀ a' b' : ℝ, line_bisects_circle a' b' → (a-2)^2 + (b-2)^2 ≤ (a'-2)^2 + (b'-2)^2) ∧
  (a-2)^2 + (b-2)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_point_l1143_114303


namespace NUMINAMATH_CALUDE_min_value_of_f_l1143_114375

noncomputable def f (x : ℝ) := x^2 + 2*x + 6/x + 9/x^2 + 4

theorem min_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = 10 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1143_114375


namespace NUMINAMATH_CALUDE_tunnel_length_l1143_114304

/-- Calculates the length of a tunnel given the train's length, speed, and time to pass through. -/
theorem tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_seconds : ℝ) :
  train_length = 300 →
  train_speed_kmh = 54 →
  time_seconds = 100 →
  (train_speed_kmh * 1000 / 3600 * time_seconds) - train_length = 1200 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_tunnel_length_l1143_114304


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l1143_114397

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 2 * Real.log x = Real.log (3 * x) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l1143_114397


namespace NUMINAMATH_CALUDE_second_stop_off_is_two_l1143_114351

/-- Represents the number of passengers on the trolley at various stages --/
structure TrolleyPassengers where
  initial : Nat
  second_stop_off : Nat
  second_stop_on : Nat
  third_stop_off : Nat
  third_stop_on : Nat
  final : Nat

/-- The trolley problem with given conditions --/
def trolleyProblem : TrolleyPassengers where
  initial := 10
  second_stop_off := 2  -- This is what we want to prove
  second_stop_on := 20  -- Twice the initial number
  third_stop_off := 18
  third_stop_on := 2
  final := 12

/-- Theorem stating that the number of people who got off at the second stop is 2 --/
theorem second_stop_off_is_two (t : TrolleyPassengers) : 
  t.initial = 10 ∧ 
  t.second_stop_on = 2 * t.initial ∧ 
  t.third_stop_off = 18 ∧ 
  t.third_stop_on = 2 ∧ 
  t.final = 12 →
  t.second_stop_off = 2 := by
  sorry

#check second_stop_off_is_two trolleyProblem

end NUMINAMATH_CALUDE_second_stop_off_is_two_l1143_114351


namespace NUMINAMATH_CALUDE_john_money_needed_l1143_114395

/-- The amount of money John currently has, in dollars -/
def current_amount : ℚ := 0.75

/-- The additional amount of money John needs, in dollars -/
def additional_amount : ℚ := 1.75

/-- The total amount of money John needs, in dollars -/
def total_amount : ℚ := current_amount + additional_amount

theorem john_money_needed : total_amount = 2.50 := by sorry

end NUMINAMATH_CALUDE_john_money_needed_l1143_114395


namespace NUMINAMATH_CALUDE_exponent_division_l1143_114398

theorem exponent_division (a : ℝ) : a ^ 3 / a = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1143_114398


namespace NUMINAMATH_CALUDE_benjamin_weekly_miles_l1143_114394

/-- Calculates the total miles Benjamin walks in a week --/
def total_miles_walked : ℕ :=
  let work_distance := 6
  let dog_walk_distance := 2
  let friend_house_distance := 1
  let store_distance := 3
  let work_days := 5
  let dog_walks_per_day := 2
  let days_in_week := 7
  let store_visits := 2
  let friend_visits := 1

  let work_miles := work_distance * 2 * work_days
  let dog_walk_miles := dog_walk_distance * dog_walks_per_day * days_in_week
  let store_miles := store_distance * 2 * store_visits
  let friend_miles := friend_house_distance * 2 * friend_visits

  work_miles + dog_walk_miles + store_miles + friend_miles

theorem benjamin_weekly_miles :
  total_miles_walked = 95 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_weekly_miles_l1143_114394


namespace NUMINAMATH_CALUDE_repunit_primes_upper_bound_l1143_114386

def repunit (k : ℕ) : ℕ := (10^k - 1) / 9

def is_repunit_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ ∃ k, repunit k = n

theorem repunit_primes_upper_bound :
  (∃ (S : Finset ℕ), ∀ n ∈ S, is_repunit_prime n ∧ n < 10^29) →
  (∃ (S : Finset ℕ), ∀ n ∈ S, is_repunit_prime n ∧ n < 10^29 ∧ S.card ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_repunit_primes_upper_bound_l1143_114386


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_l1143_114354

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

theorem f_inequality_solution_set :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_l1143_114354


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1143_114364

-- Define the universe U
def U : Set ℝ := { x | x > -3 }

-- Define set A
def A : Set ℝ := { x | x < -2 ∨ x > 3 }

-- Define set B
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem intersection_complement_equality :
  A ∩ (U \ B) = { x | -3 < x ∧ x < -2 ∨ x > 4 } := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1143_114364


namespace NUMINAMATH_CALUDE_tony_age_is_six_l1143_114350

/-- Represents Tony's work and payment details -/
structure TonyWork where
  hoursPerDay : ℕ
  payPerHourPerYear : ℚ
  daysWorked : ℕ
  totalEarned : ℚ

/-- Calculates Tony's age at the beginning of the work period -/
def calculateAge (work : TonyWork) : ℕ :=
  sorry

/-- Theorem stating that Tony's calculated age is 6 -/
theorem tony_age_is_six (work : TonyWork) 
  (h1 : work.hoursPerDay = 3)
  (h2 : work.payPerHourPerYear = 3/4)
  (h3 : work.daysWorked = 60)
  (h4 : work.totalEarned = 945) : 
  calculateAge work = 6 :=
sorry

end NUMINAMATH_CALUDE_tony_age_is_six_l1143_114350


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l1143_114344

theorem tracy_art_fair_sales (total_customers : ℕ) (first_group : ℕ) (second_group : ℕ) (third_group : ℕ)
  (second_group_paintings : ℕ) (third_group_paintings : ℕ) (total_paintings_sold : ℕ)
  (h1 : total_customers = first_group + second_group + third_group)
  (h2 : total_customers = 20)
  (h3 : first_group = 4)
  (h4 : second_group = 12)
  (h5 : third_group = 4)
  (h6 : second_group_paintings = 1)
  (h7 : third_group_paintings = 4)
  (h8 : total_paintings_sold = 36) :
  (total_paintings_sold - (second_group * second_group_paintings + third_group * third_group_paintings)) / first_group = 2 :=
sorry

end NUMINAMATH_CALUDE_tracy_art_fair_sales_l1143_114344


namespace NUMINAMATH_CALUDE_size_relationship_l1143_114308

theorem size_relationship (a b c : ℝ) : 
  a = (1/2)^(2/3) → b = (1/5)^(2/3) → c = (1/2)^(1/3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l1143_114308


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1143_114335

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = x^3 / 3 + 1 / (3 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1143_114335


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1143_114385

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ∈ ({0, 1, 2} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1143_114385


namespace NUMINAMATH_CALUDE_angle_WYZ_measure_l1143_114319

-- Define the angle measures
def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100

-- Define the theorem
theorem angle_WYZ_measure :
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 30 := by sorry

end NUMINAMATH_CALUDE_angle_WYZ_measure_l1143_114319


namespace NUMINAMATH_CALUDE_complex_product_real_l1143_114321

theorem complex_product_real (a : ℝ) :
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a : ℂ) - 2 * Complex.I) * (3 + Complex.I)).im = 0 ↔ a = 6 :=
sorry

end NUMINAMATH_CALUDE_complex_product_real_l1143_114321


namespace NUMINAMATH_CALUDE_last_two_digits_of_fraction_l1143_114312

theorem last_two_digits_of_fraction (n : ℕ) : n = 50 * 52 * 54 * 56 * 58 * 60 →
  n / 8000 ≡ 22 [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_fraction_l1143_114312


namespace NUMINAMATH_CALUDE_expression_simplification_l1143_114352

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (((x^2 - 3) / (x + 2) - x + 2) / ((x^2 - 4) / (x^2 + 4*x + 4))) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1143_114352


namespace NUMINAMATH_CALUDE_similar_polygons_area_perimeter_ratio_l1143_114359

theorem similar_polygons_area_perimeter_ratio :
  ∀ (A₁ A₂ P₁ P₂ : ℝ),
    A₁ > 0 → A₂ > 0 → P₁ > 0 → P₂ > 0 →
    (A₁ / A₂ = 9 / 64) →
    (P₁ / P₂)^2 = (A₁ / A₂) →
    P₁ / P₂ = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_similar_polygons_area_perimeter_ratio_l1143_114359


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l1143_114343

/-- Represents the number of male athletes in a stratified sample -/
def male_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (male_athletes * sample_size) / total_athletes

/-- Theorem: In a stratified sampling of 28 athletes from a team of 98 athletes (56 male and 42 female),
    the number of male athletes in the sample should be 16. -/
theorem stratified_sampling_male_athletes :
  male_athletes_in_sample 98 56 28 = 16 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l1143_114343


namespace NUMINAMATH_CALUDE_quadratic_root_difference_ratio_l1143_114380

/-- Given quadratic functions and the differences of their roots, prove the ratio of differences of squared root differences -/
theorem quadratic_root_difference_ratio (a b : ℝ) :
  let f₁ : ℝ → ℝ := λ x ↦ x^2 + 2*x + a
  let f₂ : ℝ → ℝ := λ x ↦ x^2 + b*x - 1
  let f₃ : ℝ → ℝ := λ x ↦ 2*x^2 + (6-b)*x + 3*a + 1
  let f₄ : ℝ → ℝ := λ x ↦ 2*x^2 + (3*b-2)*x - a - 3
  let A := (Real.sqrt (4 - 4*a))
  let B := (Real.sqrt (b^2 + 4))
  let C := (1/2 * Real.sqrt (b^2 - 12*b - 24*a + 28))
  let D := (1/2 * Real.sqrt (9*b^2 - 12*b + 8*a + 28))
  A ≠ B →
  (C^2 - D^2) / (A^2 - B^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_ratio_l1143_114380


namespace NUMINAMATH_CALUDE_burger_share_inches_l1143_114338

-- Define the length of the burger in feet
def burger_length_feet : ℝ := 1

-- Define the number of people sharing the burger
def num_people : ℕ := 2

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem to prove
theorem burger_share_inches : 
  (burger_length_feet * feet_to_inches) / num_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_burger_share_inches_l1143_114338


namespace NUMINAMATH_CALUDE_turtle_time_to_watering_hole_l1143_114327

/-- Represents the scenario of two lion cubs and a turtle moving towards a watering hole --/
structure WateringHoleScenario where
  /-- Speed of the first lion cub (in distance units per minute) --/
  speed_lion1 : ℝ
  /-- Distance of the first lion cub from the watering hole (in minutes) --/
  distance_lion1 : ℝ
  /-- Speed multiplier of the second lion cub relative to the first --/
  speed_multiplier_lion2 : ℝ
  /-- Distance of the turtle from the watering hole (in minutes) --/
  distance_turtle : ℝ

/-- Theorem stating the time it takes for the turtle to reach the watering hole after meeting the lion cubs --/
theorem turtle_time_to_watering_hole (scenario : WateringHoleScenario)
  (h1 : scenario.distance_lion1 = 5)
  (h2 : scenario.speed_multiplier_lion2 = 1.5)
  (h3 : scenario.distance_turtle = 30)
  (h4 : scenario.speed_lion1 > 0) :
  let meeting_time := 2
  let turtle_speed := 1 / scenario.distance_turtle
  let remaining_distance := 1 - meeting_time * turtle_speed
  remaining_distance * scenario.distance_turtle = 28 := by
  sorry

end NUMINAMATH_CALUDE_turtle_time_to_watering_hole_l1143_114327


namespace NUMINAMATH_CALUDE_cupcake_packages_l1143_114347

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : 
  initial_cupcakes = 50 → eaten_cupcakes = 5 → cupcakes_per_package = 5 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l1143_114347


namespace NUMINAMATH_CALUDE_binary_quadratic_equation_value_l1143_114305

/-- Represents a binary quadratic equation in x and y with a constant m -/
def binary_quadratic_equation (x y m : ℝ) : Prop :=
  x^2 + 2*x*y + 8*y^2 + 14*y + m = 0

/-- Represents that an equation is equivalent to two lines -/
def represents_two_lines (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), ∀ x y m,
    f x y m ↔ (a₁*x + b₁*y + c₁ = 0 ∧ a₂*x + b₂*y + c₂ = 0)

theorem binary_quadratic_equation_value :
  represents_two_lines binary_quadratic_equation → ∃ m, ∀ x y, binary_quadratic_equation x y m :=
by
  sorry

end NUMINAMATH_CALUDE_binary_quadratic_equation_value_l1143_114305


namespace NUMINAMATH_CALUDE_sequence_exists_l1143_114330

def is_valid_sequence (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, seq n + seq (n + 1) + seq (n + 2) = 15

def is_repeating (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, seq n = seq (n + 3)

theorem sequence_exists : ∃ seq : ℕ → ℕ, is_valid_sequence seq ∧ is_repeating seq :=
sorry

end NUMINAMATH_CALUDE_sequence_exists_l1143_114330


namespace NUMINAMATH_CALUDE_original_price_of_tv_l1143_114301

/-- The original price of a television given a discount and total paid amount -/
theorem original_price_of_tv (discount_rate : ℚ) (total_paid : ℚ) : 
  discount_rate = 5 / 100 → 
  total_paid = 456 → 
  (1 - discount_rate) * 480 = total_paid :=
by sorry

end NUMINAMATH_CALUDE_original_price_of_tv_l1143_114301


namespace NUMINAMATH_CALUDE_zacks_marbles_l1143_114382

theorem zacks_marbles (friends : ℕ) (ratio : List ℕ) (leftover : ℕ) (initial : ℕ) :
  friends = 9 →
  ratio = [5, 6, 7, 8, 9, 10, 11, 12, 13] →
  leftover = 27 →
  initial = (ratio.sum * 3) + leftover →
  initial = 270 :=
by sorry

end NUMINAMATH_CALUDE_zacks_marbles_l1143_114382


namespace NUMINAMATH_CALUDE_logical_propositions_l1143_114355

theorem logical_propositions (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  (((¬p) → ¬(p ∨ q)) ∧ ¬((p ∨ q) → ¬(¬p))) := by
  sorry

end NUMINAMATH_CALUDE_logical_propositions_l1143_114355


namespace NUMINAMATH_CALUDE_indeterminate_existence_l1143_114336

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Q : U → Prop)  -- Q(x) means x is a quadrilateral
variable (A : U → Prop)  -- A(x) means x has property A

-- State the theorem
theorem indeterminate_existence (h : ¬(∀ x, Q x → A x)) :
  ¬(∀ p q : Prop, p = (∃ x, Q x ∧ A x) → (q = True ∨ q = False)) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_existence_l1143_114336


namespace NUMINAMATH_CALUDE_mixture_weight_is_3_64_l1143_114315

-- Define the weights of brands in grams per liter
def weight_a : ℚ := 950
def weight_b : ℚ := 850

-- Define the ratio of volumes
def ratio_a : ℚ := 3
def ratio_b : ℚ := 2

-- Define the total volume in liters
def total_volume : ℚ := 4

-- Define the function to calculate the weight of the mixture in kg
def mixture_weight : ℚ :=
  ((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a +
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000

-- Theorem statement
theorem mixture_weight_is_3_64 : mixture_weight = 3.64 := by
  sorry

end NUMINAMATH_CALUDE_mixture_weight_is_3_64_l1143_114315


namespace NUMINAMATH_CALUDE_worker_savings_percentage_l1143_114373

theorem worker_savings_percentage
  (last_year_salary : ℝ)
  (last_year_savings_percentage : ℝ)
  (this_year_salary_increase : ℝ)
  (this_year_savings_percentage : ℝ)
  (h1 : this_year_salary_increase = 0.20)
  (h2 : this_year_savings_percentage = 0.05)
  (h3 : this_year_savings_percentage * (1 + this_year_salary_increase) * last_year_salary = last_year_savings_percentage * last_year_salary)
  : last_year_savings_percentage = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_percentage_l1143_114373


namespace NUMINAMATH_CALUDE_trajectory_is_line_with_equal_tangents_l1143_114369

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 4
def circle_O2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

-- Define the trajectory
def trajectory (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 - 4 = (x - 3)^2 + (y - 2)^2 - 1

-- Define tangent length squared to O1
def tangent_length_sq_O1 (x y : ℝ) : ℝ := (x + 1)^2 + (y + 1)^2 - 4

-- Define tangent length squared to O2
def tangent_length_sq_O2 (x y : ℝ) : ℝ := (x - 3)^2 + (y - 2)^2 - 1

-- Theorem statement
theorem trajectory_is_line_with_equal_tangents :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, trajectory x y ↔ a * x + b * y + c = 0) ∧
    (∀ x y : ℝ, trajectory x y → tangent_length_sq_O1 x y = tangent_length_sq_O2 x y) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_line_with_equal_tangents_l1143_114369


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l1143_114328

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 20)
  (h2 : selling_price = 25) :
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l1143_114328


namespace NUMINAMATH_CALUDE_curve_C_and_perpendicular_lines_l1143_114302

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.1^2 = P.2

-- Define the curve C
def curve_C (M : ℝ × ℝ) : Prop := M.1^2 = 4 * M.2

-- Define the relationship between P, D, and M
def point_relationship (P D M : ℝ × ℝ) : Prop :=
  D.1 = 0 ∧ D.2 = P.2 ∧ M.1 = 2 * P.1 ∧ M.2 = P.2

-- Define the line l
def line_l (y : ℝ) : Prop := y = -1

-- Define point F
def point_F : ℝ × ℝ := (0, 1)

-- Define perpendicular lines
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem curve_C_and_perpendicular_lines :
  ∀ (P D M A B A1 B1 : ℝ × ℝ),
    parabola P →
    point_relationship P D M →
    curve_C A ∧ curve_C B →
    line_l A1.2 ∧ line_l B1.2 →
    A1.1 = A.1 ∧ B1.1 = B.1 →
    perpendicular (A1.1 - point_F.1, A1.2 - point_F.2) (B1.1 - point_F.1, B1.2 - point_F.2) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_and_perpendicular_lines_l1143_114302


namespace NUMINAMATH_CALUDE_expression_value_l1143_114317

theorem expression_value (a b : ℝ) 
  (h1 : |a| ≠ |b|) 
  (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) : 
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18/7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1143_114317


namespace NUMINAMATH_CALUDE_garden_view_theorem_l1143_114341

/-- Represents a circular garden with trees -/
structure Garden where
  radius : ℝ
  treeGridSideLength : ℝ
  treeRadius : ℝ

/-- Checks if the view from the gazebo is obstructed -/
def isViewObstructed (g : Garden) : Prop :=
  ∀ θ : ℝ, ∃ (x y : ℤ), (x : ℝ)^2 + (y : ℝ)^2 ≤ g.radius^2 ∧
    ((x : ℝ) - g.treeRadius * Real.cos θ)^2 + ((y : ℝ) - g.treeRadius * Real.sin θ)^2 ≤ g.treeGridSideLength^2

theorem garden_view_theorem (g : Garden) (h1 : g.radius = 50) (h2 : g.treeGridSideLength = 1) :
  (g.treeRadius < 1 / Real.sqrt 2501 → ¬ isViewObstructed g) ∧
  (g.treeRadius = 1 / 50 → isViewObstructed g) := by
  sorry

#check garden_view_theorem

end NUMINAMATH_CALUDE_garden_view_theorem_l1143_114341


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1143_114396

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1143_114396


namespace NUMINAMATH_CALUDE_vector_operation_l1143_114307

/-- Given vectors a and b in R², prove that 2a - b equals the expected result. -/
theorem vector_operation (a b : Fin 2 → ℝ) (h1 : a = ![2, 1]) (h2 : b = ![-3, 4]) :
  (2 • a) - b = ![7, -2] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1143_114307


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l1143_114340

/-- Prove that the length of the platform is equal to the length of the train -/
theorem train_platform_length_equality 
  (train_speed : ℝ) 
  (crossing_time : ℝ) 
  (train_length : ℝ) 
  (h1 : train_speed = 90 * 1000 / 60) -- 90 km/hr converted to m/min
  (h2 : crossing_time = 1) -- 1 minute
  (h3 : train_length = 750) -- 750 meters
  : train_length = train_speed * crossing_time - train_length := by
  sorry

end NUMINAMATH_CALUDE_train_platform_length_equality_l1143_114340


namespace NUMINAMATH_CALUDE_nested_even_function_is_even_l1143_114370

-- Define an even function
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- State the theorem
theorem nested_even_function_is_even
  (g : ℝ → ℝ)
  (h : even_function g) :
  even_function (fun x ↦ g (g (g (g x)))) :=
by sorry

end NUMINAMATH_CALUDE_nested_even_function_is_even_l1143_114370


namespace NUMINAMATH_CALUDE_wheel_speed_is_seven_l1143_114324

noncomputable def wheel_speed (circumference : Real) (r : Real) : Prop :=
  let miles_per_rotation := circumference / 5280
  let t := miles_per_rotation / r
  let new_t := t - 1 / (3 * 3600)
  (r + 3) * new_t = miles_per_rotation

theorem wheel_speed_is_seven :
  ∀ (r : Real),
    wheel_speed 15 r →
    r = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_speed_is_seven_l1143_114324


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1143_114399

-- Define the vectors a and b
def a (x : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => 4
  | 2 => x

def b (y : ℝ) : Fin 3 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => y
  | 2 => 2

-- Theorem statement
theorem parallel_vectors_sum (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 3, a x i = k * b y i)) →
  x + y = 6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1143_114399


namespace NUMINAMATH_CALUDE_intersection_point_circle_tangent_to_l₃_l1143_114300

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

-- Define the intersection point C
def C : ℝ × ℝ := (-2, 4)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 4)^2 = 9

-- Theorem 1: Prove that C is the intersection of l₁ and l₂
theorem intersection_point : l₁ C.1 C.2 ∧ l₂ C.1 C.2 := by sorry

-- Theorem 2: Prove that the circle equation represents a circle with center C and tangent to l₃
theorem circle_tangent_to_l₃ : 
  ∃ (r : ℝ), r > 0 ∧ 
  (∀ (x y : ℝ), circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
  (∃ (x y : ℝ), l₃ x y ∧ circle_equation x y) ∧
  (∀ (x y : ℝ), l₃ x y → (x - C.1)^2 + (y - C.2)^2 ≥ r^2) := by sorry

end NUMINAMATH_CALUDE_intersection_point_circle_tangent_to_l₃_l1143_114300


namespace NUMINAMATH_CALUDE_last_three_digits_of_6_to_150_l1143_114306

theorem last_three_digits_of_6_to_150 :
  6^150 % 1000 = 126 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_6_to_150_l1143_114306


namespace NUMINAMATH_CALUDE_infinitely_many_special_prisms_l1143_114388

/-- A rectangular prism with two equal edges and the third differing by 1 -/
structure SpecialPrism where
  a : ℕ
  b : ℕ
  h_b : b = a + 1 ∨ b = a - 1

/-- The body diagonal of a rectangular prism is an integer -/
def has_integer_diagonal (p : SpecialPrism) : Prop :=
  ∃ d : ℕ, d^2 = 2 * p.a^2 + p.b^2

/-- There are infinitely many rectangular prisms with integer edges and diagonal,
    where two edges are equal and the third differs by 1 -/
theorem infinitely_many_special_prisms :
  ∀ n : ℕ, ∃ (prisms : Finset SpecialPrism),
    prisms.card > n ∧ ∀ p ∈ prisms, has_integer_diagonal p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_prisms_l1143_114388


namespace NUMINAMATH_CALUDE_triangle_segment_length_l1143_114349

/-- Given a triangle ADE with points B, C, F, and G on its sides, prove that FC = 10.25 -/
theorem triangle_segment_length (DC CB : ℝ) (h1 : DC = 9) (h2 : CB = 8)
  (AD AB ED : ℝ) (h3 : AB = (1/4) * AD) (h4 : ED = (3/4) * AD) : 
  ∃ (FC : ℝ), FC = 10.25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l1143_114349


namespace NUMINAMATH_CALUDE_sum_of_phi_plus_one_divisors_l1143_114390

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- A divisor of n is a natural number that divides n without a remainder -/
def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem sum_of_phi_plus_one_divisors (n : ℕ) :
  ∃ (divisors : Finset ℕ), 
    (∀ d ∈ divisors, is_divisor d n) ∧ 
    (Finset.card divisors = phi n + 1) ∧
    (Finset.sum divisors id = n) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_phi_plus_one_divisors_l1143_114390


namespace NUMINAMATH_CALUDE_amoeba_population_after_10_days_l1143_114381

/-- The number of amoebas after n days, given an initial population of 2 -/
def amoeba_population (n : ℕ) : ℕ := 2 * 3^n

/-- Theorem stating that the amoeba population after 10 days is 118098 -/
theorem amoeba_population_after_10_days : amoeba_population 10 = 118098 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_population_after_10_days_l1143_114381


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l1143_114361

theorem xy_greater_than_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) :
  x * y > x * z := by sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l1143_114361


namespace NUMINAMATH_CALUDE_unique_box_configuration_l1143_114372

/-- Represents a square piece --/
structure Square :=
  (id : Nat)

/-- Represents the F-shaped figure --/
def FShape := List Square

/-- Represents additional squares --/
def AdditionalSquares := List Square

/-- Represents a possible configuration of an open rectangular box --/
def BoxConfiguration := List Square

/-- A function that attempts to form an open rectangular box --/
def formBox (f : FShape) (add : AdditionalSquares) : Option BoxConfiguration :=
  sorry

/-- The main theorem stating there's exactly one valid configuration --/
theorem unique_box_configuration 
  (f : FShape) 
  (add : AdditionalSquares) 
  (h1 : f.length = 7) 
  (h2 : add.length = 3) : 
  ∃! (box : BoxConfiguration), formBox f add = some box :=
sorry

end NUMINAMATH_CALUDE_unique_box_configuration_l1143_114372


namespace NUMINAMATH_CALUDE_divisible_by_35_60_72_between_1000_and_3500_l1143_114346

theorem divisible_by_35_60_72_between_1000_and_3500 : 
  ∃! n : ℕ, 1000 < n ∧ n < 3500 ∧ 35 ∣ n ∧ 60 ∣ n ∧ 72 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_35_60_72_between_1000_and_3500_l1143_114346


namespace NUMINAMATH_CALUDE_complex_distance_theorem_l1143_114333

theorem complex_distance_theorem : ∃ (c : ℂ) (d : ℝ), c ≠ 0 ∧ 
  ∀ (z : ℂ), Complex.abs z = 1 → 1 + z + z^2 ≠ 0 → 
    Complex.abs (1 / (1 + z + z^2)) - Complex.abs (1 / (1 + z + z^2) - c) = d :=
by sorry

end NUMINAMATH_CALUDE_complex_distance_theorem_l1143_114333


namespace NUMINAMATH_CALUDE_product_of_roots_l1143_114348

theorem product_of_roots (x : ℝ) : 
  (∃ a b c : ℝ, (x + 3) * (x - 4) = 2 * (x + 1) ∧ 
   a * x^2 + b * x + c = 0 ∧ 
   (∀ r s : ℝ, (a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) → r * s = c / a)) →
  (∃ r s : ℝ, (r + 3) * (r - 4) = 2 * (r + 1) ∧ 
              (s + 3) * (s - 4) = 2 * (s + 1) ∧ 
              r * s = -14) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1143_114348


namespace NUMINAMATH_CALUDE_count_four_digit_integers_thousands_4_l1143_114377

/-- The count of four-digit positive integers with the thousands digit 4 -/
def fourDigitIntegersWithThousands4 : ℕ :=
  (Finset.range 10).card * (Finset.range 10).card * (Finset.range 10).card

/-- Theorem stating that the count of four-digit positive integers with the thousands digit 4 is 1000 -/
theorem count_four_digit_integers_thousands_4 :
  fourDigitIntegersWithThousands4 = 1000 := by sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_thousands_4_l1143_114377


namespace NUMINAMATH_CALUDE_solve_jim_ring_problem_l1143_114334

def jim_ring_problem (first_ring_cost : ℝ) : Prop :=
  let second_ring_cost : ℝ := 2 * first_ring_cost
  let sale_price : ℝ := first_ring_cost / 2
  let out_of_pocket : ℝ := first_ring_cost + second_ring_cost - sale_price
  (first_ring_cost = 10000) → (out_of_pocket = 25000)

theorem solve_jim_ring_problem :
  jim_ring_problem 10000 := by
  sorry

end NUMINAMATH_CALUDE_solve_jim_ring_problem_l1143_114334


namespace NUMINAMATH_CALUDE_inequality_property_l1143_114309

theorem inequality_property (a b c : ℝ) : 
  a / c^2 < b / c^2 → a < b := by
sorry

end NUMINAMATH_CALUDE_inequality_property_l1143_114309


namespace NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_sixth_power_l1143_114358

theorem nearest_integer_to_three_plus_sqrt_five_sixth_power :
  ⌊(3 + Real.sqrt 5)^6 + 1/2⌋ = 20608 := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_sixth_power_l1143_114358


namespace NUMINAMATH_CALUDE_initial_customers_l1143_114391

theorem initial_customers (initial leaving new final : ℕ) : 
  leaving = 8 → new = 4 → final = 9 → 
  initial - leaving + new = final → 
  initial = 13 := by sorry

end NUMINAMATH_CALUDE_initial_customers_l1143_114391


namespace NUMINAMATH_CALUDE_distribute_three_items_five_people_l1143_114368

/-- The number of ways to distribute distinct items among distinct people -/
def distribute_items (num_items : ℕ) (num_people : ℕ) : ℕ :=
  num_people ^ num_items

/-- Theorem: Distributing 3 distinct items among 5 distinct people results in 125 ways -/
theorem distribute_three_items_five_people : 
  distribute_items 3 5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_distribute_three_items_five_people_l1143_114368


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1143_114320

theorem perfect_square_binomial : ∃ (r s : ℝ), (r * x + s)^2 = 4 * x^2 + 20 * x + 25 := by sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1143_114320


namespace NUMINAMATH_CALUDE_sqrt_13_squared_l1143_114345

theorem sqrt_13_squared : (Real.sqrt 13) ^ 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_13_squared_l1143_114345


namespace NUMINAMATH_CALUDE_stream_speed_l1143_114332

/-- 
Given a boat with a speed of 22 km/hr in still water that travels 108 km downstream in 4 hours,
prove that the speed of the stream is 5 km/hr.
-/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 22 →
  distance = 108 →
  time = 4 →
  boat_speed + stream_speed = distance / time →
  stream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l1143_114332


namespace NUMINAMATH_CALUDE_increase_then_decrease_l1143_114316

theorem increase_then_decrease (x p q : ℝ) (hx : x = 80) (hp : p = 150) (hq : q = 30) :
  x * (1 + p / 100) * (1 - q / 100) = 140 := by
  sorry

end NUMINAMATH_CALUDE_increase_then_decrease_l1143_114316


namespace NUMINAMATH_CALUDE_not_right_triangle_l1143_114339

theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l1143_114339


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1143_114337

/-- Proves that a parallelogram with area 44 cm² and height 11 cm has a base of 4 cm -/
theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (is_parallelogram : Bool) 
  (h1 : is_parallelogram = true)
  (h2 : area = 44)
  (h3 : height = 11) :
  area / height = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1143_114337


namespace NUMINAMATH_CALUDE_product_sum_multiplier_l1143_114371

theorem product_sum_multiplier (a b k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : a + b = k * (a * b)) (h2 : 1 / a + 1 / b = 6) : k = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_multiplier_l1143_114371


namespace NUMINAMATH_CALUDE_range_of_k_with_two_preimages_l1143_114325

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem range_of_k_with_two_preimages :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ f x = k ∧ f y = k) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_with_two_preimages_l1143_114325


namespace NUMINAMATH_CALUDE_time_to_work_calculation_l1143_114366

-- Define the problem parameters
def speed_to_work : ℝ := 50
def speed_to_home : ℝ := 110
def total_time : ℝ := 2

-- Define the theorem
theorem time_to_work_calculation :
  ∃ (distance : ℝ) (time_to_work : ℝ),
    distance / speed_to_work + distance / speed_to_home = total_time ∧
    time_to_work = distance / speed_to_work ∧
    time_to_work * 60 = 82.5 := by
  sorry


end NUMINAMATH_CALUDE_time_to_work_calculation_l1143_114366


namespace NUMINAMATH_CALUDE_rectangle_ratio_around_square_l1143_114365

/-- Given a square surrounded by four identical rectangles, this theorem proves
    that the ratio of the longer side to the shorter side of each rectangle is 2,
    when the area of the larger square formed is 9 times that of the inner square. -/
theorem rectangle_ratio_around_square : 
  ∀ (s x y : ℝ),
  s > 0 →  -- inner square side length is positive
  x > y → y > 0 →  -- rectangle dimensions are positive and x is longer
  (s + 2*y)^2 = 9*s^2 →  -- area relation
  (x + s)^2 = 9*s^2 →  -- outer square side length
  x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_around_square_l1143_114365


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l1143_114331

theorem sequence_gcd_property (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l1143_114331


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1143_114311

theorem supplementary_angles_ratio (angle1 angle2 : ℝ) : 
  angle1 + angle2 = 180 →  -- angles are supplementary
  angle1 = 4 * angle2 →    -- angles are in ratio 4:1
  angle2 = 36 :=           -- smaller angle is 36°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1143_114311


namespace NUMINAMATH_CALUDE_commodity_price_problem_l1143_114376

theorem commodity_price_problem (price1 price2 : ℕ) : 
  price1 + price2 = 827 →
  price1 = price2 + 127 →
  price1 = 477 := by
sorry

end NUMINAMATH_CALUDE_commodity_price_problem_l1143_114376


namespace NUMINAMATH_CALUDE_number_times_fifteen_equals_150_l1143_114329

theorem number_times_fifteen_equals_150 :
  ∃ x : ℝ, 15 * x = 150 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_times_fifteen_equals_150_l1143_114329


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l1143_114314

/-- The number of factors of 12000 that are perfect squares -/
def num_perfect_square_factors : ℕ :=
  sorry

/-- 12000 expressed as its prime factorization -/
def twelve_thousand_factorization : ℕ :=
  2^5 * 3 * 5^3

theorem count_perfect_square_factors :
  num_perfect_square_factors = 6 ∧ twelve_thousand_factorization = 12000 :=
sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l1143_114314


namespace NUMINAMATH_CALUDE_final_book_count_l1143_114323

/-- The number of storybooks in a library after borrowing and returning books. -/
def library_books (initial : ℕ) (borrowed : ℕ) (returned : ℕ) : ℕ :=
  initial - borrowed + returned

/-- Theorem stating that given the initial conditions, the library ends up with 72 books. -/
theorem final_book_count :
  library_books 95 58 35 = 72 := by
  sorry

end NUMINAMATH_CALUDE_final_book_count_l1143_114323


namespace NUMINAMATH_CALUDE_complex_multiplication_equal_parts_l1143_114379

theorem complex_multiplication_equal_parts (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_multiplication_equal_parts_l1143_114379


namespace NUMINAMATH_CALUDE_numbers_solution_l1143_114389

def find_numbers (x y z : ℤ) : Prop :=
  (y = 2*x - 3) ∧ 
  (x + y = 51) ∧ 
  (z = 4*x - y)

theorem numbers_solution : 
  ∃ (x y z : ℤ), find_numbers x y z ∧ x = 18 ∧ y = 33 ∧ z = 39 :=
sorry

end NUMINAMATH_CALUDE_numbers_solution_l1143_114389


namespace NUMINAMATH_CALUDE_sqrt_sin_cos_identity_l1143_114367

theorem sqrt_sin_cos_identity (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sin_cos_identity_l1143_114367


namespace NUMINAMATH_CALUDE_perfect_square_condition_solution_uniqueness_l1143_114322

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def repeated_digits (x y : ℕ) (n : ℕ) : ℕ :=
  x * 10^(2*n) + 6 * 10^n + y

theorem perfect_square_condition (x y : ℕ) : Prop :=
  x ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → is_perfect_square (repeated_digits x y n)

theorem solution_uniqueness :
  ∀ x y : ℕ, perfect_square_condition x y →
    ((x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_solution_uniqueness_l1143_114322


namespace NUMINAMATH_CALUDE_brady_work_hours_september_l1143_114342

/-- Proves that Brady worked 8 hours every day in September given the conditions --/
theorem brady_work_hours_september :
  let hours_per_day_april : ℕ := 6
  let hours_per_day_june : ℕ := 5
  let days_per_month : ℕ := 30
  let average_hours_per_month : ℕ := 190
  let total_months : ℕ := 3
  ∃ (hours_per_day_september : ℕ),
    hours_per_day_september * days_per_month =
      total_months * average_hours_per_month -
      (hours_per_day_april * days_per_month + hours_per_day_june * days_per_month) ∧
    hours_per_day_september = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_brady_work_hours_september_l1143_114342


namespace NUMINAMATH_CALUDE_smallest_prime_triangle_perimeter_l1143_114318

/-- A triangle with prime side lengths and prime perimeter -/
structure PrimeTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  ha : Nat.Prime a
  hb : Nat.Prime b
  hc : Nat.Prime c
  hab : a < b
  hbc : b < c
  hmin : 5 ≤ a
  htri1 : a + b > c
  htri2 : a + c > b
  htri3 : b + c > a
  hperi : Nat.Prime (a + b + c)

/-- The theorem stating the smallest perimeter of a PrimeTriangle is 23 -/
theorem smallest_prime_triangle_perimeter :
  ∀ t : PrimeTriangle, 23 ≤ t.a + t.b + t.c ∧
  ∃ t0 : PrimeTriangle, t0.a + t0.b + t0.c = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_triangle_perimeter_l1143_114318


namespace NUMINAMATH_CALUDE_age_difference_of_children_l1143_114353

theorem age_difference_of_children (n : ℕ) (sum_ages : ℕ) (eldest_age : ℕ) (d : ℕ) : 
  n = 5 → 
  sum_ages = 50 → 
  eldest_age = 14 → 
  sum_ages = n * eldest_age - (d * (n * (n - 1)) / 2) → 
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_of_children_l1143_114353


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1143_114384

/-- An arithmetic sequence with sum S_n and common difference d -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  S : ℕ → ℝ
  hd : d ≠ 0
  hS : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2
  ha : ∀ n, a n = a 1 + (n - 1) * d

/-- The main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : ∀ n, seq.S n ≤ seq.S 8) : 
  seq.d < 0 ∧ seq.S 17 ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1143_114384


namespace NUMINAMATH_CALUDE_sum_of_roots_l1143_114387

theorem sum_of_roots (k c d : ℝ) (y₁ y₂ : ℝ) : 
  y₁ ≠ y₂ →
  5 * y₁^2 - k * y₁ - c = 0 →
  5 * y₂^2 - k * y₂ - c = 0 →
  5 * y₁^2 - k * y₁ = d →
  5 * y₂^2 - k * y₂ = d →
  d ≠ c →
  y₁ + y₂ = k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1143_114387


namespace NUMINAMATH_CALUDE_car_production_proof_l1143_114357

/-- The total number of cars produced over two days, given production on each day --/
def total_cars (day1 : ℕ) (day2 : ℕ) : ℕ := day1 + day2

/-- The number of cars produced on the second day is twice that of the first day --/
def double_production (day1 : ℕ) : ℕ := 2 * day1

theorem car_production_proof (day1 : ℕ) (h1 : day1 = 60) :
  total_cars day1 (double_production day1) = 180 := by
  sorry


end NUMINAMATH_CALUDE_car_production_proof_l1143_114357


namespace NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l1143_114374

/-- The smallest positive integer M such that M and M^2 + 1 end in the same sequence of four digits in base 10, where the first digit of the four is not zero. -/
def smallest_matching_end_digits : ℕ := 3125

/-- Predicate to check if a number ends with the same four digits as its square plus one. -/
def ends_with_same_four_digits (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≡ n^2 + 1 [ZMOD 10000]

theorem smallest_matching_end_digits_correct :
  ends_with_same_four_digits smallest_matching_end_digits ∧
  ∀ m : ℕ, m < smallest_matching_end_digits → ¬ends_with_same_four_digits m := by
  sorry

end NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l1143_114374


namespace NUMINAMATH_CALUDE_horner_eval_negative_two_l1143_114326

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 + 3x^4 + 2x^3 - x^2 - x - 1/2 -/
def f : List ℚ := [4, 3, 2, -1, -1, -1/2]

/-- Theorem: f(-2) = -197/2 using Horner's method -/
theorem horner_eval_negative_two :
  horner_eval f (-2) = -197/2 := by
  sorry

#eval horner_eval f (-2)

end NUMINAMATH_CALUDE_horner_eval_negative_two_l1143_114326


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1143_114392

theorem fraction_multiplication : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1143_114392


namespace NUMINAMATH_CALUDE_repeating_decimal_interval_l1143_114362

/-- A number is a repeating decimal with period p if it can be expressed as m / (10^p - 1) for some integer m. -/
def is_repeating_decimal (x : ℚ) (p : ℕ) : Prop :=
  ∃ (m : ℤ), x = m / (10^p - 1)

theorem repeating_decimal_interval :
  ∀ n : ℕ,
    n < 2000 →
    is_repeating_decimal (1 / n) 8 →
    is_repeating_decimal (1 / (n + 6)) 6 →
    801 ≤ n ∧ n ≤ 1200 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_interval_l1143_114362
