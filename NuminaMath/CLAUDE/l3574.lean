import Mathlib

namespace NUMINAMATH_CALUDE_exists_range_sum_and_even_count_l3574_357402

/-- Sum of integers from n to m, inclusive -/
def sum_range (n m : ℤ) : ℤ := (m - n + 1) * (n + m) / 2

/-- Number of even integers from n to m, inclusive -/
def count_even (n m : ℤ) : ℤ :=
  if (n % 2 = m % 2) then (m - n) / 2 + 1 else (m - n + 1) / 2

/-- Theorem stating the existence of a range satisfying the given conditions -/
theorem exists_range_sum_and_even_count :
  ∃ (n m : ℤ), n ≤ m ∧ sum_range n m + count_even n m = 641 :=
sorry

end NUMINAMATH_CALUDE_exists_range_sum_and_even_count_l3574_357402


namespace NUMINAMATH_CALUDE_f_root_exists_l3574_357489

noncomputable def f (x : ℝ) := Real.log x / Real.log 3 + x

theorem f_root_exists : ∃ x ∈ Set.Ioo 3 4, f x - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_root_exists_l3574_357489


namespace NUMINAMATH_CALUDE_oyster_consumption_l3574_357472

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- The number of oysters Crabby eats -/
def crabby_oysters : ℕ := 2 * squido_oysters

/-- The total number of oysters eaten by Crabby and Squido -/
def total_oysters : ℕ := squido_oysters + crabby_oysters

theorem oyster_consumption :
  total_oysters = 600 :=
sorry

end NUMINAMATH_CALUDE_oyster_consumption_l3574_357472


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_26_l3574_357456

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

theorem smallest_positive_integer_ending_in_9_divisible_by_26 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_9 n ∧ n % 26 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_9 m → m % 26 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_26_l3574_357456


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l3574_357449

theorem x_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, x ^ 2 - 3 * x + 2 = 0 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 - 3 * x + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_l3574_357449


namespace NUMINAMATH_CALUDE_min_value_theorem_l3574_357458

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 4/y ≥ 1/a + 4/b) →
  1/a + 4/b = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3574_357458


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_l3574_357424

theorem product_of_two_digit_numbers : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧
  10 ≤ b ∧ b < 100 ∧
  a * b = 4725 ∧
  min a b = 15 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_l3574_357424


namespace NUMINAMATH_CALUDE_right_triangle_cone_volumes_l3574_357419

/-- Given a right triangle with legs a and b, if the volume of the cone formed by
    rotating about leg a is 675π cm³ and the volume of the cone formed by rotating
    about leg b is 1215π cm³, then the length of the hypotenuse is 3√106 cm. -/
theorem right_triangle_cone_volumes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / 3 : ℝ) * π * b^2 * a = 675 * π →
  (1 / 3 : ℝ) * π * a^2 * b = 1215 * π →
  Real.sqrt (a^2 + b^2) = 3 * Real.sqrt 106 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_cone_volumes_l3574_357419


namespace NUMINAMATH_CALUDE_cauchy_equation_solution_l3574_357446

theorem cauchy_equation_solution (f : ℝ → ℝ) 
  (h_cauchy : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (h_condition : f 1 ^ 2 = f 1) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_equation_solution_l3574_357446


namespace NUMINAMATH_CALUDE_min_value_3a_plus_b_min_value_exists_min_value_equality_l3574_357470

theorem min_value_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  ∀ x y, x > 0 → y > 0 → x + 2*y = x*y → 3*a + b ≤ 3*x + y :=
by sorry

theorem min_value_exists (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = x*y ∧ 3*x + y = 7 + 2*Real.sqrt 6 :=
by sorry

theorem min_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = a*b) :
  3*a + b ≥ 7 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3a_plus_b_min_value_exists_min_value_equality_l3574_357470


namespace NUMINAMATH_CALUDE_planes_lines_false_implications_l3574_357486

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

theorem planes_lines_false_implications 
  (α β : Plane) (l m : Line) :
  ∃ (α β : Plane) (l m : Line),
    α ≠ β ∧ l ≠ m ∧
    subset l α ∧ subset m β ∧
    ¬(¬(parallel α β) → ¬(line_parallel l m)) ∧
    ¬(perpendicular l m → plane_perpendicular α β) := by
  sorry

end NUMINAMATH_CALUDE_planes_lines_false_implications_l3574_357486


namespace NUMINAMATH_CALUDE_g_continuous_c_plus_d_equals_negative_three_l3574_357498

-- Define the piecewise function g(x)
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if -3 ≤ x ∧ x ≤ 1 then 2 * x - 4
  else 3 * x - d

-- Theorem stating the continuity condition
theorem g_continuous (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) ↔ c = -4 ∧ d = 1 := by
  sorry

-- Corollary for the sum of c and d
theorem c_plus_d_equals_negative_three (c d : ℝ) :
  (∀ x, ContinuousAt (g c d) x) → c + d = -3 := by
  sorry

end NUMINAMATH_CALUDE_g_continuous_c_plus_d_equals_negative_three_l3574_357498


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l3574_357466

/-- A parabola defined by the equation y = -x² + 2x + m --/
def parabola (x y m : ℝ) : Prop := y = -x^2 + 2*x + m

/-- Point A on the parabola --/
def point_A (y₁ m : ℝ) : Prop := parabola (-1) y₁ m

/-- Point B on the parabola --/
def point_B (y₂ m : ℝ) : Prop := parabola 1 y₂ m

/-- Point C on the parabola --/
def point_C (y₃ m : ℝ) : Prop := parabola 2 y₃ m

theorem parabola_point_relationship (y₁ y₂ y₃ m : ℝ) 
  (hA : point_A y₁ m) (hB : point_B y₂ m) (hC : point_C y₃ m) : 
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l3574_357466


namespace NUMINAMATH_CALUDE_total_books_calculation_l3574_357417

theorem total_books_calculation (joan_books tom_books lisa_books steve_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : lisa_books = 27)
  (h4 : steve_books = 45) :
  joan_books + tom_books + lisa_books + steve_books = 120 := by
sorry

end NUMINAMATH_CALUDE_total_books_calculation_l3574_357417


namespace NUMINAMATH_CALUDE_computable_logarithms_l3574_357462

def is_computable (n : ℕ) : Prop :=
  ∃ (m n p : ℕ), n = 2^m * 3^n * 5^p ∧ n ≤ 100

def computable_set : Set ℕ :=
  {n : ℕ | n ≥ 1 ∧ n ≤ 100 ∧ is_computable n}

theorem computable_logarithms :
  computable_set = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100} :=
by sorry

end NUMINAMATH_CALUDE_computable_logarithms_l3574_357462


namespace NUMINAMATH_CALUDE_series_general_term_l3574_357473

theorem series_general_term (n : ℕ) (a : ℕ → ℚ) :
  (∀ k, a k = 1 / (k^2 : ℚ)) →
  a n = 1 / (n^2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_series_general_term_l3574_357473


namespace NUMINAMATH_CALUDE_product_equals_three_l3574_357495

theorem product_equals_three (a b c d : ℚ) 
  (ha : a + 3 = 3 * a)
  (hb : b + 4 = 4 * b)
  (hc : c + 5 = 5 * c)
  (hd : d + 6 = 6 * d) : 
  a * b * c * d = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_three_l3574_357495


namespace NUMINAMATH_CALUDE_no_correct_letter_probability_l3574_357423

/-- The number of people and envelopes --/
def n : ℕ := 7

/-- The factorial function --/
def factorial (k : ℕ) : ℕ := if k = 0 then 1 else k * factorial (k - 1)

/-- The derangement function --/
def derangement (k : ℕ) : ℕ := 
  if k = 0 then 1
  else if k = 1 then 0
  else (k - 1) * (derangement (k - 1) + derangement (k - 2))

/-- Theorem: The probability of no one receiving their correct letter 
    when n letters are randomly distributed to n people is equal to 427/1160 --/
theorem no_correct_letter_probability : 
  (derangement n : ℚ) / (factorial n : ℚ) = 427 / 1160 := by sorry

end NUMINAMATH_CALUDE_no_correct_letter_probability_l3574_357423


namespace NUMINAMATH_CALUDE_haley_carrots_count_l3574_357448

/-- The number of carrots Haley picked -/
def haley_carrots : ℕ := 39

/-- The number of carrots Haley's mom picked -/
def mom_carrots : ℕ := 38

/-- The number of good carrots -/
def good_carrots : ℕ := 64

/-- The number of bad carrots -/
def bad_carrots : ℕ := 13

theorem haley_carrots_count : haley_carrots = 39 := by
  have total_carrots : ℕ := good_carrots + bad_carrots
  have total_carrots_alt : ℕ := haley_carrots + mom_carrots
  have h1 : total_carrots = total_carrots_alt := by sorry
  sorry

end NUMINAMATH_CALUDE_haley_carrots_count_l3574_357448


namespace NUMINAMATH_CALUDE_greatest_integer_third_side_l3574_357480

theorem greatest_integer_third_side (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), x > c → ¬(x < a + b ∧ x > |a - b| ∧ a < b + x ∧ b < a + x)) ∧
  (c < a + b ∧ c > |a - b| ∧ a < b + c ∧ b < a + c) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_third_side_l3574_357480


namespace NUMINAMATH_CALUDE_triangle_sum_maximum_l3574_357441

theorem triangle_sum_maximum (arrangement : List ℕ) : 
  (arrangement.toFinset = Finset.range 9 \ {0}) →
  (∃ (side1 side2 side3 : List ℕ), 
    side1.length = 4 ∧ side2.length = 4 ∧ side3.length = 4 ∧
    (side1 ++ side2 ++ side3).toFinset = arrangement.toFinset ∧
    side1.sum = side2.sum ∧ side2.sum = side3.sum) →
  (∀ (side : List ℕ), side.toFinset ⊆ arrangement.toFinset ∧ side.length = 4 → side.sum ≤ 19) :=
by sorry

#check triangle_sum_maximum

end NUMINAMATH_CALUDE_triangle_sum_maximum_l3574_357441


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3574_357414

/-- A quadratic function passing through (1,0) and (-3,0) with minimum value 25 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_one : a + b + c = 0
  passes_through_neg_three : 9*a - 3*b + c = 0
  has_minimum_25 : ∀ x, a*x^2 + b*x + c ≥ 25

/-- The sum of coefficients a + b + c equals -75/4 for the given quadratic function -/
theorem quadratic_sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = -75/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3574_357414


namespace NUMINAMATH_CALUDE_specific_room_surface_area_l3574_357416

/-- Calculates the interior surface area of a cubic room with a central cubical hole -/
def interior_surface_area (room_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  6 * room_edge^2 - 3 * hole_edge^2

/-- Theorem stating the interior surface area of a specific cubic room with a hole -/
theorem specific_room_surface_area :
  interior_surface_area 10 2 = 588 := by
  sorry

#check specific_room_surface_area

end NUMINAMATH_CALUDE_specific_room_surface_area_l3574_357416


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3574_357453

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h1 : a * b = 62216)
  (h2 : Nat.gcd a b = 22) : 
  Nat.lcm a b = 2828 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3574_357453


namespace NUMINAMATH_CALUDE_two_true_propositions_l3574_357468

theorem two_true_propositions : 
  let original := ∀ a : ℝ, a > 2 → a > 1
  let converse := ∀ a : ℝ, a > 1 → a > 2
  let inverse := ∀ a : ℝ, a ≤ 2 → a ≤ 1
  let contrapositive := ∀ a : ℝ, a ≤ 1 → a ≤ 2
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3574_357468


namespace NUMINAMATH_CALUDE_negate_all_men_are_good_drivers_l3574_357463

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Man : U → Prop)
variable (GoodDriver : U → Prop)

-- Define the statements
def AllMenAreGoodDrivers : Prop := ∀ x, Man x → GoodDriver x
def AtLeastOneManIsBadDriver : Prop := ∃ x, Man x ∧ ¬GoodDriver x

-- Theorem to prove
theorem negate_all_men_are_good_drivers :
  AtLeastOneManIsBadDriver U Man GoodDriver ↔ ¬(AllMenAreGoodDrivers U Man GoodDriver) :=
sorry

end NUMINAMATH_CALUDE_negate_all_men_are_good_drivers_l3574_357463


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3574_357464

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (h1 : a 2 = 12)  -- second term is 12
  (h2 : a 6 = 4)   -- sixth term is 4
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3574_357464


namespace NUMINAMATH_CALUDE_final_a_is_three_l3574_357400

/-- Given initial values of a and b, compute the final value of a after the operation a = a + b -/
def compute_final_a (initial_a : ℕ) (initial_b : ℕ) : ℕ :=
  initial_a + initial_b

/-- Theorem stating that given the initial conditions a = 1 and b = 2, 
    after the operation a = a + b, the final value of a is 3 -/
theorem final_a_is_three : compute_final_a 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_final_a_is_three_l3574_357400


namespace NUMINAMATH_CALUDE_total_students_is_184_l3574_357443

/-- Represents the number of students that can be transported in one car for a school --/
structure CarCapacity where
  capacity : ℕ

/-- Represents a school participating in the competition --/
structure School where
  students : ℕ
  carCapacity : CarCapacity

/-- Represents the state of both schools at a given point --/
structure CompetitionState where
  school1 : School
  school2 : School

/-- Checks if the given state satisfies the initial conditions --/
def initialConditionsSatisfied (state : CompetitionState) : Prop :=
  state.school1.students = state.school2.students ∧
  state.school1.carCapacity.capacity = 15 ∧
  state.school2.carCapacity.capacity = 13 ∧
  (state.school2.students + state.school2.carCapacity.capacity - 1) / state.school2.carCapacity.capacity =
    (state.school1.students / state.school1.carCapacity.capacity) + 1

/-- Checks if the given state satisfies the conditions after adding one student to each school --/
def middleConditionsSatisfied (state : CompetitionState) : Prop :=
  (state.school1.students + 1) / state.school1.carCapacity.capacity =
  (state.school2.students + 1) / state.school2.carCapacity.capacity

/-- Checks if the given state satisfies the final conditions --/
def finalConditionsSatisfied (state : CompetitionState) : Prop :=
  ((state.school1.students + 2) / state.school1.carCapacity.capacity) + 1 =
  (state.school2.students + 2) / state.school2.carCapacity.capacity

/-- The main theorem stating that under the given conditions, the total number of students is 184 --/
theorem total_students_is_184 (state : CompetitionState) :
  initialConditionsSatisfied state →
  middleConditionsSatisfied state →
  finalConditionsSatisfied state →
  state.school1.students + state.school2.students + 4 = 184 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_is_184_l3574_357443


namespace NUMINAMATH_CALUDE_max_temperature_range_l3574_357403

theorem max_temperature_range (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 50)
  (min_temp : ∃ i, temps i = 45 ∧ ∀ j, temps j ≥ 45) :
  ∃ i j, temps i - temps j ≤ 25 ∧ 
  ∀ k l, temps k - temps l ≤ temps i - temps j :=
by sorry

end NUMINAMATH_CALUDE_max_temperature_range_l3574_357403


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l3574_357428

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l3574_357428


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3574_357406

def TwoStepSelection (n : ℕ) (m : ℕ) (k : ℕ) :=
  (n > m) ∧ (m > k) ∧ (k > 0)

theorem equal_selection_probability
  (n m k : ℕ)
  (h : TwoStepSelection n m k)
  (eliminate_one : ℕ → ℕ)
  (systematic_sample : ℕ → Finset ℕ)
  (h_eliminate : ∀ i, i ∈ Finset.range n → eliminate_one i ∈ Finset.range (n - 1))
  (h_sample : ∀ i, i ∈ Finset.range (n - 1) → systematic_sample i ⊆ Finset.range (n - 1) ∧ (systematic_sample i).card = k) :
  ∀ j ∈ Finset.range n, (∃ i ∈ Finset.range (n - 1), j ∈ systematic_sample (eliminate_one i)) ↔ true :=
sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l3574_357406


namespace NUMINAMATH_CALUDE_puppies_given_away_l3574_357454

def initial_puppies : ℕ := 7
def remaining_puppies : ℕ := 2

theorem puppies_given_away : initial_puppies - remaining_puppies = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_away_l3574_357454


namespace NUMINAMATH_CALUDE_fayes_remaining_money_fayes_remaining_money_is_30_l3574_357415

/-- Calculates the remaining money for Faye after receiving money from her mother and making purchases. -/
theorem fayes_remaining_money (initial_money : ℝ) (cupcake_price : ℝ) (cupcake_quantity : ℕ) 
  (cookie_box_price : ℝ) (cookie_box_quantity : ℕ) : ℝ :=
  let mother_gift := 2 * initial_money
  let total_money := initial_money + mother_gift
  let cupcake_cost := cupcake_price * cupcake_quantity
  let cookie_cost := cookie_box_price * cookie_box_quantity
  let total_spent := cupcake_cost + cookie_cost
  total_money - total_spent

/-- Proves that Faye's remaining money is $30 given the initial conditions. -/
theorem fayes_remaining_money_is_30 : 
  fayes_remaining_money 20 1.5 10 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fayes_remaining_money_fayes_remaining_money_is_30_l3574_357415


namespace NUMINAMATH_CALUDE_fifth_hour_speed_l3574_357429

def speed_hour1 : ℝ := 90
def speed_hour2 : ℝ := 60
def speed_hour3 : ℝ := 120
def speed_hour4 : ℝ := 72
def average_speed : ℝ := 80
def total_time : ℝ := 5

def total_distance : ℝ := average_speed * total_time

def distance_first_four_hours : ℝ := speed_hour1 + speed_hour2 + speed_hour3 + speed_hour4

def speed_hour5 : ℝ := total_distance - distance_first_four_hours

theorem fifth_hour_speed :
  speed_hour5 = 58 := by sorry

end NUMINAMATH_CALUDE_fifth_hour_speed_l3574_357429


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l3574_357434

/-- Represents the inverse variation relationship between 5y and x^3 -/
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, 5 * y = k / (x^3)

theorem inverse_variation_solution :
  ∀ f : ℝ → ℝ,
  (∀ x, inverse_variation x (f x)) →  -- Condition: 5y varies inversely as the cube of x
  f 2 = 4 →                           -- Condition: When y = 4, x = 2
  f 4 = 1/2                           -- Conclusion: y = 1/2 when x = 4
:= by sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l3574_357434


namespace NUMINAMATH_CALUDE_simplify_expression_l3574_357413

theorem simplify_expression : (6 + 6 + 12) / 3 - 2 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3574_357413


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3574_357481

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3574_357481


namespace NUMINAMATH_CALUDE_sum_equals_eight_l3574_357412

theorem sum_equals_eight (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq1 : b * (a + b + c) + a * c ≥ 16)
  (h_ineq2 : a + 2 * b + c ≤ 8) : 
  a + 2 * b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_eight_l3574_357412


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3574_357490

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 1 ↔ -2 < x ∧ x < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3574_357490


namespace NUMINAMATH_CALUDE_count_not_divisible_by_5_and_7_l3574_357408

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  n - (n / a + n / b - n / (a * b))

theorem count_not_divisible_by_5_and_7 :
  count_not_divisible 499 5 7 = 343 := by sorry

end NUMINAMATH_CALUDE_count_not_divisible_by_5_and_7_l3574_357408


namespace NUMINAMATH_CALUDE_exactly_two_cheaper_to_buy_more_l3574_357450

-- Define the cost function
def C (n : ℕ) : ℝ :=
  if n ≤ 30 then 15 * n - 20
  else if n ≤ 55 then 14 * n
  else 13 * n + 10

-- Define a function that checks if it's cheaper to buy n+1 books than n books
def cheaperToBuyMore (n : ℕ) : Prop := C (n + 1) < C n

-- Theorem statement
theorem exactly_two_cheaper_to_buy_more :
  ∃! (s : Finset ℕ), (∀ n ∈ s, cheaperToBuyMore n) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_cheaper_to_buy_more_l3574_357450


namespace NUMINAMATH_CALUDE_unique_sum_value_l3574_357426

theorem unique_sum_value (n m : ℤ) 
  (h1 : 3 * n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3 * m - 2 * n < 46) :
  2 * n + m = 36 := by
sorry

end NUMINAMATH_CALUDE_unique_sum_value_l3574_357426


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l3574_357485

theorem largest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 2 * c →
  5 * a = 2 * d →
  d = 480 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l3574_357485


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l3574_357422

theorem cube_face_perimeter (volume : ℝ) (perimeter : ℝ) : 
  volume = 512 → perimeter = 32 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l3574_357422


namespace NUMINAMATH_CALUDE_bruce_fruit_purchase_l3574_357478

/-- Calculates the total amount paid for fruits given their quantities and rates. -/
def totalAmountPaid (grapeQuantity mangoQuantity grapeRate mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Proves that Bruce paid 1055 to the shopkeeper for his fruit purchase. -/
theorem bruce_fruit_purchase : totalAmountPaid 8 9 70 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_purchase_l3574_357478


namespace NUMINAMATH_CALUDE_athletes_with_four_points_after_seven_rounds_l3574_357418

/-- The number of athletes with k points after m rounds in a tournament of 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n-m) * (m.choose k)

/-- The total number of athletes with 4 points after 7 rounds in a tournament of 2^n + 6 participants -/
def athletes_with_four_points (n : ℕ) : ℕ := 35 * 2^(n-7) + 2

theorem athletes_with_four_points_after_seven_rounds (n : ℕ) (h : n > 7) :
  athletes_with_four_points n = f n 7 4 + 2 :=
sorry

#check athletes_with_four_points_after_seven_rounds

end NUMINAMATH_CALUDE_athletes_with_four_points_after_seven_rounds_l3574_357418


namespace NUMINAMATH_CALUDE_remaining_space_is_7200_mb_l3574_357436

/-- Conversion factor from GB to MB -/
def gb_to_mb : ℕ := 1024

/-- Total hard drive capacity in GB -/
def total_capacity_gb : ℕ := 300

/-- Used storage space in MB -/
def used_space_mb : ℕ := 300000

/-- Theorem: The remaining empty space on the hard drive is 7200 MB -/
theorem remaining_space_is_7200_mb :
  total_capacity_gb * gb_to_mb - used_space_mb = 7200 := by
  sorry

end NUMINAMATH_CALUDE_remaining_space_is_7200_mb_l3574_357436


namespace NUMINAMATH_CALUDE_total_eyes_is_92_l3574_357483

/-- Represents a monster family in the portrait --/
structure MonsterFamily where
  totalEyes : ℕ

/-- The main monster family --/
def mainFamily : MonsterFamily :=
  { totalEyes := 1 + 3 + 3 * 4 + 5 + 6 + 2 + 1 + 7 + 8 }

/-- The first neighboring monster family --/
def neighborFamily1 : MonsterFamily :=
  { totalEyes := 9 + 3 + 7 + 3 }

/-- The second neighboring monster family --/
def neighborFamily2 : MonsterFamily :=
  { totalEyes := 4 + 2 * 8 + 5 }

/-- The total number of eyes in the monster family portrait --/
def totalEyesInPortrait : ℕ :=
  mainFamily.totalEyes + neighborFamily1.totalEyes + neighborFamily2.totalEyes

/-- Theorem stating that the total number of eyes in the portrait is 92 --/
theorem total_eyes_is_92 : totalEyesInPortrait = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_is_92_l3574_357483


namespace NUMINAMATH_CALUDE_fraction_product_minus_one_l3574_357476

theorem fraction_product_minus_one : 
  (2/3) * (3/4) * (4/5) * (5/6) * (6/7) * (7/8) * (8/9) - 1 = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_minus_one_l3574_357476


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l3574_357479

def power_product : ℕ := 2^2010 * 5^2008 * 7

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_power_product : sum_of_digits power_product = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l3574_357479


namespace NUMINAMATH_CALUDE_line_through_point_l3574_357465

/-- A line contains a point if the point's coordinates satisfy the line's equation. -/
def line_contains_point (m : ℚ) (x y : ℚ) : Prop :=
  2 - m * x = -4 * y

/-- The theorem states that the line 2 - mx = -4y contains the point (5, -2) when m = -6/5. -/
theorem line_through_point (m : ℚ) :
  line_contains_point m 5 (-2) ↔ m = -6/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l3574_357465


namespace NUMINAMATH_CALUDE_bicycle_sale_loss_percentage_l3574_357496

theorem bicycle_sale_loss_percentage
  (profit_A_to_B : ℝ)
  (profit_A_to_C : ℝ)
  (h1 : profit_A_to_B = 0.30)
  (h2 : profit_A_to_C = 0.040000000000000036) :
  ∃ (loss_B_to_C : ℝ), loss_B_to_C = 0.20 ∧ 
    (1 + profit_A_to_C) = (1 + profit_A_to_B) * (1 - loss_B_to_C) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_sale_loss_percentage_l3574_357496


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3574_357420

theorem arithmetic_expression_equality : (24 / (8 + 2 - 6)) * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3574_357420


namespace NUMINAMATH_CALUDE_max_value_theorem_l3574_357467

theorem max_value_theorem (a b c : ℝ) (h : a^2 + b^2 + c^2 ≤ 8) :
  4 * (a^3 + b^3 + c^3) - (a^4 + b^4 + c^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3574_357467


namespace NUMINAMATH_CALUDE_symmetric_lines_l3574_357471

/-- Given two lines L and K symmetric to each other with respect to y=x,
    where L has equation y = ax + b (a ≠ 0, b ≠ 0),
    prove that K has equation y = (1/a)x - (b/a) -/
theorem symmetric_lines (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let L : ℝ → ℝ := fun x => a * x + b
  let K : ℝ → ℝ := fun x => (1 / a) * x - (b / a)
  (∀ x y, y = L x ↔ x = L y) →
  (∀ x, K x = (1 / a) * x - (b / a)) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_l3574_357471


namespace NUMINAMATH_CALUDE_sara_letters_count_l3574_357482

/-- The number of letters Sara sent in January -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_count :
  total_letters = 33 := by sorry

end NUMINAMATH_CALUDE_sara_letters_count_l3574_357482


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3574_357444

/-- Calculates the final amount after two years of compound interest with different rates each year. -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated. -/
theorem compound_interest_calculation :
  final_amount 5460 0.04 0.05 = 5962.32 := by
  sorry

#eval final_amount 5460 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l3574_357444


namespace NUMINAMATH_CALUDE_square_division_theorem_l3574_357457

theorem square_division_theorem (x : ℝ) (h1 : x > 0) :
  (∃ l : ℝ, l > 0 ∧ 2 * l = x^2 / 5) →
  (∃ a : ℝ, a > 0 ∧ x * a = x^2 / 5) →
  x = 8 ∧ x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_division_theorem_l3574_357457


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3574_357487

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_cond : 2 * a 1 + a 2 = a 3) :
  (a 4 + a 5) / (a 3 + a 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3574_357487


namespace NUMINAMATH_CALUDE_egg_box_count_l3574_357401

theorem egg_box_count (total_eggs : Real) (eggs_per_box : Real) (h1 : total_eggs = 3.0) (h2 : eggs_per_box = 1.5) :
  (total_eggs / eggs_per_box : Real) = 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_box_count_l3574_357401


namespace NUMINAMATH_CALUDE_probability_two_females_selected_l3574_357445

/-- The probability of selecting 2 females out of 6 finalists (4 females and 2 males) -/
theorem probability_two_females_selected (total : Nat) (females : Nat) (selected : Nat) 
  (h1 : total = 6) 
  (h2 : females = 4)
  (h3 : selected = 2) : 
  (Nat.choose females selected : ℚ) / (Nat.choose total selected) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_selected_l3574_357445


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3574_357460

/-- Definition of the sequence of pairs -/
def pair_sequence : ℕ → ℕ × ℕ
| n => let group := (n + 1).sqrt
       let position := n - (group * (group - 1)) / 2
       (position, group + 1 - position)

/-- Theorem stating that the 60th pair is (5, 7) -/
theorem sixtieth_pair_is_five_seven :
  pair_sequence 60 = (5, 7) := by
  sorry


end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3574_357460


namespace NUMINAMATH_CALUDE_unique_solution_system_l3574_357461

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 - 23*y - 25*z = -681) ∧
  (y^2 - 21*x - 21*z = -419) ∧
  (z^2 - 19*x - 21*y = -313) ↔
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3574_357461


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3574_357492

theorem arithmetic_sequence_common_difference
  (a : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, a n + a (n + 1) = 4 * n)
  : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ+, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3574_357492


namespace NUMINAMATH_CALUDE_probability_three_blue_jellybeans_l3574_357493

def total_jellybeans : ℕ := 20
def initial_blue_jellybeans : ℕ := 10

def probability_all_blue : ℚ := 2 / 19

theorem probability_three_blue_jellybeans :
  let p1 := initial_blue_jellybeans / total_jellybeans
  let p2 := (initial_blue_jellybeans - 1) / (total_jellybeans - 1)
  let p3 := (initial_blue_jellybeans - 2) / (total_jellybeans - 2)
  p1 * p2 * p3 = probability_all_blue := by
  sorry

end NUMINAMATH_CALUDE_probability_three_blue_jellybeans_l3574_357493


namespace NUMINAMATH_CALUDE_factorization_proof_l3574_357410

theorem factorization_proof (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3574_357410


namespace NUMINAMATH_CALUDE_triangle_side_length_l3574_357452

theorem triangle_side_length (a b : ℝ) (C : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : C = 2 * π / 3) :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3574_357452


namespace NUMINAMATH_CALUDE_dolphin_edge_probability_l3574_357404

/-- The probability of a point being within 2 m of the edge in a 30 m by 20 m rectangle is 23/75. -/
theorem dolphin_edge_probability : 
  let pool_length : ℝ := 30
  let pool_width : ℝ := 20
  let edge_distance : ℝ := 2
  let total_area := pool_length * pool_width
  let inner_length := pool_length - 2 * edge_distance
  let inner_width := pool_width - 2 * edge_distance
  let inner_area := inner_length * inner_width
  let edge_area := total_area - inner_area
  edge_area / total_area = 23 / 75 := by sorry

end NUMINAMATH_CALUDE_dolphin_edge_probability_l3574_357404


namespace NUMINAMATH_CALUDE_unique_tuple_l3574_357488

def satisfies_condition (a : Fin 9 → ℕ+) : Prop :=
  ∀ i j k l, i < j → j < k → k ≤ 9 → l ≠ i → l ≠ j → l ≠ k → l ≤ 9 →
    a i + a j + a k + a l = 100

theorem unique_tuple : ∃! a : Fin 9 → ℕ+, satisfies_condition a := by
  sorry

end NUMINAMATH_CALUDE_unique_tuple_l3574_357488


namespace NUMINAMATH_CALUDE_water_needed_proof_l3574_357475

/-- The ratio of water to lemon juice in the lemonade recipe -/
def water_ratio : ℚ := 8 / 10

/-- The number of gallons of lemonade to make -/
def gallons_to_make : ℚ := 2

/-- The number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- The number of liters in a quart -/
def liters_per_quart : ℚ := 95 / 100

/-- The amount of water needed in liters -/
def water_needed : ℚ := 
  water_ratio * gallons_to_make * quarts_per_gallon * liters_per_quart

theorem water_needed_proof : water_needed = 608 / 100 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_proof_l3574_357475


namespace NUMINAMATH_CALUDE_arrange_programs_count_l3574_357407

/-- The number of ways to arrange n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 5 programs with 2 consecutive -/
def arrange_programs : ℕ :=
  2 * permutations 4

theorem arrange_programs_count : arrange_programs = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrange_programs_count_l3574_357407


namespace NUMINAMATH_CALUDE_parallelogram_base_l3574_357477

/-- The base of a parallelogram given its area and height -/
theorem parallelogram_base (area height base : ℝ) 
  (h_area : area = 384) 
  (h_height : height = 16) 
  (h_formula : area = base * height) : 
  base = 24 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3574_357477


namespace NUMINAMATH_CALUDE_units_digit_problem_l3574_357425

theorem units_digit_problem : ∃ n : ℕ, (6 * 16 * 1986 - 6^4) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l3574_357425


namespace NUMINAMATH_CALUDE_harry_change_problem_l3574_357474

theorem harry_change_problem (change : ℕ) : 
  change < 100 ∧ 
  change % 50 = 2 ∧ 
  change % 5 = 4 → 
  change = 52 := by sorry

end NUMINAMATH_CALUDE_harry_change_problem_l3574_357474


namespace NUMINAMATH_CALUDE_trig_identity_l3574_357433

theorem trig_identity (α : Real) (h : Real.sin (α - π/12) = 1/3) : 
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3574_357433


namespace NUMINAMATH_CALUDE_inscribed_circle_cycle_l3574_357494

structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def inscribed_circle (T : Triangle) (i : ℕ) : Circle :=
  sorry

theorem inscribed_circle_cycle (T : Triangle) :
  inscribed_circle T 7 = inscribed_circle T 1 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_cycle_l3574_357494


namespace NUMINAMATH_CALUDE_triangle_sum_l3574_357435

theorem triangle_sum (AC BC : ℝ) (HE HD : ℝ) (a b : ℝ) :
  AC = 16.25 →
  BC = 13.75 →
  HE = 6 →
  HD = 3 →
  b - a = 5 →
  BC * (HD + b) = AC * (HE + a) →
  a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_l3574_357435


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3574_357491

/-- The coefficient of x^4 in the expansion of (1 + √x)^10 -/
def coefficient_x4 : ℕ :=
  Nat.choose 10 8

theorem expansion_coefficient (n : ℕ) :
  coefficient_x4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3574_357491


namespace NUMINAMATH_CALUDE_consecutive_integers_prime_divisor_ratio_l3574_357437

theorem consecutive_integers_prime_divisor_ratio :
  ∃ a : ℕ, ∀ i ∈ Finset.range 2009,
    let n := a + i + 1
    ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧
      (∀ r : ℕ, Prime r → r ∣ n → p ≤ r) ∧
      (∀ r : ℕ, Prime r → r ∣ n → r ≤ q) ∧
      q > 20 * p :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_prime_divisor_ratio_l3574_357437


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l3574_357484

/-- The time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time (jogger_speed train_speed : ℝ) 
  (initial_distance train_length : ℝ) (h1 : jogger_speed > 0) 
  (h2 : train_speed > jogger_speed) (h3 : initial_distance > 0) 
  (h4 : train_length > 0) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) = 40 := by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l3574_357484


namespace NUMINAMATH_CALUDE_songs_leftover_l3574_357409

theorem songs_leftover (total_songs : ℕ) (num_playlists : ℕ) (h1 : total_songs = 2048) (h2 : num_playlists = 13) :
  total_songs % num_playlists = 7 := by
  sorry

end NUMINAMATH_CALUDE_songs_leftover_l3574_357409


namespace NUMINAMATH_CALUDE_wendy_album_problem_l3574_357440

/-- Given a total number of pictures and the number of pictures in each of 5 albums,
    calculate the number of pictures in the first album. -/
def pictures_in_first_album (total : ℕ) (pics_per_album : ℕ) : ℕ :=
  total - 5 * pics_per_album

/-- Theorem stating that given 79 total pictures and 7 pictures in each of 5 albums,
    the number of pictures in the first album is 44. -/
theorem wendy_album_problem :
  pictures_in_first_album 79 7 = 44 := by
  sorry

end NUMINAMATH_CALUDE_wendy_album_problem_l3574_357440


namespace NUMINAMATH_CALUDE_largest_integer_in_fraction_inequality_five_satisfies_inequality_largest_integer_is_five_l3574_357421

theorem largest_integer_in_fraction_inequality :
  ∀ x : ℤ, (2 : ℚ) / 5 < (x : ℚ) / 7 ∧ (x : ℚ) / 7 < 8 / 11 → x ≤ 5 :=
by sorry

theorem five_satisfies_inequality :
  (2 : ℚ) / 5 < (5 : ℚ) / 7 ∧ (5 : ℚ) / 7 < 8 / 11 :=
by sorry

theorem largest_integer_is_five :
  ∃! x : ℤ, x = 5 ∧
    ((2 : ℚ) / 5 < (x : ℚ) / 7 ∧ (x : ℚ) / 7 < 8 / 11) ∧
    (∀ y : ℤ, (2 : ℚ) / 5 < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < 8 / 11 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_fraction_inequality_five_satisfies_inequality_largest_integer_is_five_l3574_357421


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3574_357455

def n : ℕ := 81 * 83 * 85 * 87 + 89

theorem distinct_prime_factors_count : Nat.card (Nat.factors n).toFinset = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3574_357455


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3574_357442

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3574_357442


namespace NUMINAMATH_CALUDE_cake_division_possible_l3574_357497

/-- Represents the different ways a cake can be divided -/
inductive CakePortion
  | Whole
  | Half
  | Third

/-- Represents the distribution of cakes to children -/
structure CakeDistribution where
  whole : Nat
  half : Nat
  third : Nat

/-- Calculates the total portion of cake for a given distribution -/
def totalPortion (d : CakeDistribution) : Rat :=
  d.whole + d.half / 2 + d.third / 3

theorem cake_division_possible : ∃ (d : CakeDistribution),
  -- Each child gets the same amount
  totalPortion d = 13 / 6 ∧
  -- The distribution uses exactly 13 cakes
  d.whole + d.half + d.third = 13 ∧
  -- The number of half cakes is even (so they can be paired)
  d.half % 2 = 0 ∧
  -- The number of third cakes is divisible by 3 (so they can be grouped)
  d.third % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_cake_division_possible_l3574_357497


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_2531_l3574_357430

theorem largest_prime_factor_of_2531 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2531 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2531 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_2531_l3574_357430


namespace NUMINAMATH_CALUDE_difference_of_largest_prime_factors_l3574_357447

theorem difference_of_largest_prime_factors : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ p * q = 172081 ∧ 
  ∀ (r : Nat), Nat.Prime r ∧ r ∣ 172081 → r ≤ max p q ∧
  max p q - min p q = 13224 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_largest_prime_factors_l3574_357447


namespace NUMINAMATH_CALUDE_roots_of_x_squared_equals_x_l3574_357431

theorem roots_of_x_squared_equals_x :
  {x : ℝ | x^2 = x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_roots_of_x_squared_equals_x_l3574_357431


namespace NUMINAMATH_CALUDE_distance_from_origin_l3574_357432

theorem distance_from_origin (x y : ℝ) (h1 : y = 15) 
  (h2 : Real.sqrt ((x - 2)^2 + (y - 8)^2) = 13) (h3 : x > 2) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3574_357432


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3574_357405

theorem consecutive_integers_sum (a b c : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c = 13) → a + b + c = 36 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3574_357405


namespace NUMINAMATH_CALUDE_scientific_notation_of_280000_l3574_357438

theorem scientific_notation_of_280000 :
  280000 = 2.8 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_280000_l3574_357438


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l3574_357427

/-- Given points A, B, C, D on a line segment AB where AB = 4AD and AB = 8BC,
    prove that the probability of a randomly selected point on AB
    being between C and D is 5/8. -/
theorem probability_between_C_and_D (A B C D : ℝ) : 
  A < C ∧ C < D ∧ D < B →  -- Points are in order on the line segment
  B - A = 4 * (D - A) →    -- AB = 4AD
  B - A = 8 * (C - B) →    -- AB = 8BC
  (D - C) / (B - A) = 5/8 := by sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l3574_357427


namespace NUMINAMATH_CALUDE_distance_between_trees_problem_l3574_357459

/-- The distance between consecutive trees in a yard -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  (yard_length : ℚ) / (num_trees - 1 : ℚ)

/-- Theorem: The distance between consecutive trees in a 400-meter yard with 26 trees is 16 meters -/
theorem distance_between_trees_problem :
  distance_between_trees 400 26 = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_problem_l3574_357459


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_zero_l3574_357439

theorem sum_of_solutions_equals_zero : 
  ∃ (S : Finset Int), 
    (∀ x ∈ S, x^4 - 13*x^2 + 36 = 0) ∧ 
    (∀ x : Int, x^4 - 13*x^2 + 36 = 0 → x ∈ S) ∧ 
    (S.sum id = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equals_zero_l3574_357439


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l3574_357411

theorem distance_from_origin_to_point :
  let x : ℝ := 8
  let y : ℝ := -15
  (x^2 + y^2).sqrt = 17 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l3574_357411


namespace NUMINAMATH_CALUDE_julia_puppy_cost_l3574_357499

def adoption_fee : ℝ := 20
def dog_food : ℝ := 20
def treats_price : ℝ := 2.5
def treats_quantity : ℕ := 2
def toys : ℝ := 15
def crate : ℝ := 20
def bed : ℝ := 20
def collar_leash : ℝ := 15
def discount_rate : ℝ := 0.2

def total_cost : ℝ :=
  adoption_fee +
  (1 - discount_rate) * (dog_food + treats_price * treats_quantity + toys + crate + bed + collar_leash)

theorem julia_puppy_cost :
  total_cost = 96 := by sorry

end NUMINAMATH_CALUDE_julia_puppy_cost_l3574_357499


namespace NUMINAMATH_CALUDE_load_truck_time_proof_l3574_357469

/-- The time taken for three workers to load one truck simultaneously -/
def time_to_load_truck (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating the time taken for the given workers to load one truck -/
theorem load_truck_time_proof :
  let rate1 : ℚ := 1 / 5
  let rate2 : ℚ := 1 / 4
  let rate3 : ℚ := 1 / 6
  time_to_load_truck rate1 rate2 rate3 = 60 / 37 := by
  sorry

#eval time_to_load_truck (1/5) (1/4) (1/6)

end NUMINAMATH_CALUDE_load_truck_time_proof_l3574_357469


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_2_pow_12_l3574_357451

-- Define the function to get the last digit of a rational number's decimal expansion
noncomputable def lastDigitOfDecimalExpansion (q : ℚ) : ℕ :=
  sorry

-- Theorem statement
theorem last_digit_of_one_over_2_pow_12 :
  lastDigitOfDecimalExpansion (1 / 2^12) = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_2_pow_12_l3574_357451
