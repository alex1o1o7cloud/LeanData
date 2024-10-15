import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l294_29449

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l294_29449


namespace NUMINAMATH_CALUDE_students_liking_both_pizza_and_burgers_l294_29485

theorem students_liking_both_pizza_and_burgers 
  (total : ℕ) 
  (pizza : ℕ) 
  (burgers : ℕ) 
  (neither : ℕ) 
  (h1 : total = 50) 
  (h2 : pizza = 22) 
  (h3 : burgers = 20) 
  (h4 : neither = 14) : 
  pizza + burgers - (total - neither) = 6 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_pizza_and_burgers_l294_29485


namespace NUMINAMATH_CALUDE_time_conversion_l294_29436

-- Define the conversion rates
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Define the given time
def hours : ℕ := 3
def minutes : ℕ := 25

-- Theorem to prove
theorem time_conversion :
  (hours * minutes_per_hour + minutes) * seconds_per_minute = 12300 := by
  sorry

end NUMINAMATH_CALUDE_time_conversion_l294_29436


namespace NUMINAMATH_CALUDE_library_book_redistribution_l294_29457

theorem library_book_redistribution (total_boxes : Nat) (books_per_box : Nat) (new_box_capacity : Nat) :
  total_boxes = 1421 →
  books_per_box = 27 →
  new_box_capacity = 35 →
  (total_boxes * books_per_box) % new_box_capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_library_book_redistribution_l294_29457


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l294_29438

theorem arithmetic_mean_geq_geometric_mean (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l294_29438


namespace NUMINAMATH_CALUDE_simplify_fraction_l294_29450

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) :
  (1 - 1 / x) / ((1 - x^2) / x) = -1 / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l294_29450


namespace NUMINAMATH_CALUDE_unique_set_A_l294_29483

def M : Set ℤ := {1, 3, 5, 7, 9}

theorem unique_set_A : ∃! A : Set ℤ, A.Nonempty ∧ 
  (∀ a ∈ A, a + 4 ∈ M) ∧ 
  (∀ a ∈ A, a - 4 ∈ M) ∧
  A = {5} := by sorry

end NUMINAMATH_CALUDE_unique_set_A_l294_29483


namespace NUMINAMATH_CALUDE_original_denominator_problem_l294_29459

theorem original_denominator_problem (d : ℕ) : 
  (3 : ℚ) / d ≠ 0 → 
  (6 : ℚ) / (d + 3) = 1 / 3 → 
  d = 15 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l294_29459


namespace NUMINAMATH_CALUDE_vector_operation_proof_l294_29425

theorem vector_operation_proof :
  (4 : ℝ) • (![3, -5] : Fin 2 → ℝ) - (3 : ℝ) • (![2, -6] : Fin 2 → ℝ) = ![6, -2] := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l294_29425


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_two_l294_29413

/-- The opposite of a real number -/
def opposite (a : ℝ) : ℝ := -a

/-- The property that defines the opposite of a number -/
theorem opposite_def (a : ℝ) : a + opposite a = 0 := by sorry

/-- Proof that the opposite of -2 is 2 -/
theorem opposite_of_neg_two : opposite (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_two_l294_29413


namespace NUMINAMATH_CALUDE_cost_of_one_each_l294_29455

/-- The cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions from the problem -/
def problem_conditions (cost : GoodsCost) : Prop :=
  3 * cost.A + 7 * cost.B + cost.C = 3.15 ∧
  4 * cost.A + 10 * cost.B + cost.C = 4.20

/-- The theorem to prove -/
theorem cost_of_one_each (cost : GoodsCost) :
  problem_conditions cost → cost.A + cost.B + cost.C = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_each_l294_29455


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l294_29404

open Real

-- Define the statements p and q
def p : Prop := ∀ x, (deriv (λ x => 3 * x^2 + Real.log 3)) x = 6 * x + 3

def q : Prop := ∀ x, x ∈ Set.Ioo (-3 : ℝ) 1 ↔ 
  (deriv (λ x => (3 - x^2) * Real.exp x)) x > 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l294_29404


namespace NUMINAMATH_CALUDE_f_properties_l294_29444

def f (x : ℝ) := x^2

theorem f_properties : 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l294_29444


namespace NUMINAMATH_CALUDE_zeros_of_f_l294_29473

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem zeros_of_f :
  {x : ℝ | f x = 0} = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l294_29473


namespace NUMINAMATH_CALUDE_equal_fish_count_l294_29441

def herring_fat : ℕ := 40
def eel_fat : ℕ := 20
def pike_fat : ℕ := eel_fat + 10
def total_fat : ℕ := 3600

theorem equal_fish_count (x : ℕ) 
  (h : x * herring_fat + x * eel_fat + x * pike_fat = total_fat) : 
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_fish_count_l294_29441


namespace NUMINAMATH_CALUDE_right_triangle_existence_l294_29472

theorem right_triangle_existence (a q : ℝ) (ha : a > 0) (hq : q > 0) :
  ∃ (b c : ℝ), 
    b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    (b^2 / c) = q :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l294_29472


namespace NUMINAMATH_CALUDE_line_intersect_xz_plane_l294_29499

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_intersect_xz_plane (p₁ p₂ intersection : ℝ × ℝ × ℝ) :
  p₁ = (1, 2, 3) →
  p₂ = (4, 0, -1) →
  intersection = (4, 0, -1) →
  (∃ t : ℝ, intersection = p₁ + t • (p₂ - p₁)) ∧
  (intersection.2 = 0) := by
  sorry

#check line_intersect_xz_plane

end NUMINAMATH_CALUDE_line_intersect_xz_plane_l294_29499


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l294_29410

/-- Given two vectors a and b in ℝ², where a = (1, 2) and b = (x, -2),
    if a and b are perpendicular, then x = 4. -/
theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -2]
  (∀ i, i < 2 → a i * b i = 0) →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l294_29410


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_207_l294_29428

theorem sum_of_last_two_digits_of_9_pow_207 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 9^207 ≡ 10*a + b [ZMOD 100] ∧ a + b = 15 :=
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_207_l294_29428


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l294_29447

theorem subtraction_multiplication_equality : 10111 - 10 * 2 * 5 = 10011 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l294_29447


namespace NUMINAMATH_CALUDE_combined_weight_l294_29442

/-- The combined weight of Tracy, John, and Jake is 150 kg -/
theorem combined_weight (tracy_weight : ℕ) (jake_weight : ℕ) (john_weight : ℕ)
  (h1 : tracy_weight = 52)
  (h2 : jake_weight = tracy_weight + 8)
  (h3 : jake_weight - john_weight = 14 ∨ tracy_weight - john_weight = 14) :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

#check combined_weight

end NUMINAMATH_CALUDE_combined_weight_l294_29442


namespace NUMINAMATH_CALUDE_loan_balance_after_ten_months_l294_29433

/-- Represents a loan with monthly payments -/
structure Loan where
  monthly_payment : ℕ
  total_months : ℕ
  current_balance : ℕ

/-- Calculates the remaining balance of a loan after a given number of months -/
def remaining_balance (loan : Loan) (months : ℕ) : ℕ :=
  loan.current_balance - loan.monthly_payment * months

/-- Theorem: Given a loan where $10 is paid back monthly, and half of the loan has been repaid 
    after 6 months, the remaining balance after 10 months will be $20 -/
theorem loan_balance_after_ten_months 
  (loan : Loan)
  (h1 : loan.monthly_payment = 10)
  (h2 : loan.total_months = 6)
  (h3 : loan.current_balance = loan.monthly_payment * loan.total_months) :
  remaining_balance loan 4 = 20 := by
  sorry


end NUMINAMATH_CALUDE_loan_balance_after_ten_months_l294_29433


namespace NUMINAMATH_CALUDE_beetles_consumed_1080_l294_29406

/-- Represents the daily consumption and population changes in a tropical forest ecosystem --/
structure ForestEcosystem where
  beetles_per_bird : ℕ
  birds_per_snake : ℕ
  snakes_per_jaguar : ℕ
  jaguars_per_crocodile : ℕ
  bird_increase : ℕ
  snake_increase : ℕ
  jaguar_increase : ℕ
  initial_jaguars : ℕ
  initial_crocodiles : ℕ

/-- Calculates the number of beetles consumed in one day in the forest ecosystem --/
def beetles_consumed (eco : ForestEcosystem) : ℕ :=
  eco.initial_jaguars * eco.snakes_per_jaguar * eco.birds_per_snake * eco.beetles_per_bird

/-- Theorem stating that the number of beetles consumed in one day is 1080 --/
theorem beetles_consumed_1080 (eco : ForestEcosystem) 
  (h1 : eco.beetles_per_bird = 12)
  (h2 : eco.birds_per_snake = 3)
  (h3 : eco.snakes_per_jaguar = 5)
  (h4 : eco.jaguars_per_crocodile = 2)
  (h5 : eco.bird_increase = 4)
  (h6 : eco.snake_increase = 2)
  (h7 : eco.jaguar_increase = 1)
  (h8 : eco.initial_jaguars = 6)
  (h9 : eco.initial_crocodiles = 30) :
  beetles_consumed eco = 1080 := by
  sorry


end NUMINAMATH_CALUDE_beetles_consumed_1080_l294_29406


namespace NUMINAMATH_CALUDE_first_discount_percentage_l294_29421

theorem first_discount_percentage (original_price final_price : ℝ) (second_discount : ℝ) :
  original_price = 26.67 →
  final_price = 15 →
  second_discount = 25 →
  ∃ first_discount : ℝ,
    first_discount = 25 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l294_29421


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l294_29405

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence where a_1 * a_5 = a_3, the value of a_3 is 1. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : IsGeometricSequence a) 
  (h_prop : a 1 * a 5 = a 3) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l294_29405


namespace NUMINAMATH_CALUDE_evaluate_expression_l294_29478

theorem evaluate_expression (a x : ℝ) (h : x = a + 9) : x - a + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l294_29478


namespace NUMINAMATH_CALUDE_relationship_p_q_l294_29489

theorem relationship_p_q (k : ℝ) : ∃ (p₀ q₀ p₁ : ℝ),
  (p₀ * q₀^2 = k) ∧ 
  (p₀ = 16) ∧ 
  (q₀ = 4) ∧ 
  (p₁ * 8^2 = k) → 
  p₁ = 4 := by
sorry

end NUMINAMATH_CALUDE_relationship_p_q_l294_29489


namespace NUMINAMATH_CALUDE_percentage_subtracted_l294_29487

theorem percentage_subtracted (a : ℝ) (h : ∃ p : ℝ, a - p * a = 0.97 * a) : 
  ∃ p : ℝ, p = 0.03 ∧ a - p * a = 0.97 * a :=
sorry

end NUMINAMATH_CALUDE_percentage_subtracted_l294_29487


namespace NUMINAMATH_CALUDE_visitors_to_both_countries_l294_29429

theorem visitors_to_both_countries 
  (total : ℕ) 
  (iceland : ℕ) 
  (norway : ℕ) 
  (neither : ℕ) 
  (h1 : total = 60) 
  (h2 : iceland = 35) 
  (h3 : norway = 23) 
  (h4 : neither = 33) : 
  ∃ (both : ℕ), both = 31 ∧ 
    total = iceland + norway - both + neither :=
by sorry

end NUMINAMATH_CALUDE_visitors_to_both_countries_l294_29429


namespace NUMINAMATH_CALUDE_initial_roses_count_l294_29464

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 3

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 12

/-- The number of roses after adding flowers -/
def final_roses : ℕ := 11

/-- The number of orchids after adding flowers -/
def final_orchids : ℕ := 20

/-- The difference between orchids and roses after adding flowers -/
def orchid_rose_difference : ℕ := 9

theorem initial_roses_count :
  initial_roses = 3 ∧
  initial_orchids = 12 ∧
  final_roses = 11 ∧
  final_orchids = 20 ∧
  orchid_rose_difference = 9 ∧
  final_orchids - final_roses = orchid_rose_difference ∧
  final_orchids - initial_orchids = final_roses - initial_roses :=
by sorry

end NUMINAMATH_CALUDE_initial_roses_count_l294_29464


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l294_29481

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l294_29481


namespace NUMINAMATH_CALUDE_system_solution_form_l294_29443

theorem system_solution_form (x y : ℝ) : 
  x + 2*y = 5 → 4*x*y = 9 → 
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    ((x = (a + b * Real.sqrt c) / d) ∨ (x = (a - b * Real.sqrt c) / d)) ∧
    a = 5 ∧ b = 1 ∧ c = 7 ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_form_l294_29443


namespace NUMINAMATH_CALUDE_kids_ticket_price_l294_29414

/-- Proves that the price of a kid's ticket is $12 given the specified conditions --/
theorem kids_ticket_price (total_people : ℕ) (adult_price : ℕ) (total_sales : ℕ) (num_kids : ℕ) :
  total_people = 254 →
  adult_price = 28 →
  total_sales = 3864 →
  num_kids = 203 →
  ∃ (kids_price : ℕ), kids_price = 12 ∧ 
    total_sales = (total_people - num_kids) * adult_price + num_kids * kids_price :=
by sorry


end NUMINAMATH_CALUDE_kids_ticket_price_l294_29414


namespace NUMINAMATH_CALUDE_number_of_B_l294_29460

/-- Given that the number of A is x and the number of B is a less than half of A,
    prove that the number of B is equal to (1/2)x - a. -/
theorem number_of_B (x a : ℝ) (hA : x ≥ 0) (hB : x ≥ 2 * a) :
  (1/2 : ℝ) * x - a = (1/2 : ℝ) * x - a :=
by sorry

end NUMINAMATH_CALUDE_number_of_B_l294_29460


namespace NUMINAMATH_CALUDE_fraction_five_seventeenths_repetend_l294_29408

/-- The repetend of a rational number in its decimal representation -/
def repetend (n d : ℕ) : List ℕ := sorry

/-- The length of the repetend of a rational number in its decimal representation -/
def repetendLength (n d : ℕ) : ℕ := sorry

theorem fraction_five_seventeenths_repetend :
  repetend 5 17 = [2, 9, 4, 1, 1, 7, 6] ∧ repetendLength 5 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_five_seventeenths_repetend_l294_29408


namespace NUMINAMATH_CALUDE_factor_implies_s_value_l294_29411

theorem factor_implies_s_value (m s : ℝ) : 
  (m - 8) ∣ (m^2 - s*m - 24) → s = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_s_value_l294_29411


namespace NUMINAMATH_CALUDE_polynomial_equality_l294_29471

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b = (x - 1)*(x + 4)) → a = 3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l294_29471


namespace NUMINAMATH_CALUDE_chord_length_l294_29463

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l294_29463


namespace NUMINAMATH_CALUDE_vacation_cost_problem_l294_29467

/-- The vacation cost problem -/
theorem vacation_cost_problem 
  (alice_paid bob_paid carol_paid dave_paid : ℚ)
  (h_alice : alice_paid = 160)
  (h_bob : bob_paid = 120)
  (h_carol : carol_paid = 140)
  (h_dave : dave_paid = 200)
  (a b : ℚ) 
  (h_equal_split : (alice_paid + bob_paid + carol_paid + dave_paid) / 4 = 
                   alice_paid - a)
  (h_bob_contribution : (alice_paid + bob_paid + carol_paid + dave_paid) / 4 = 
                        bob_paid + b) :
  a - b = -35 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_problem_l294_29467


namespace NUMINAMATH_CALUDE_lemon_heads_distribution_l294_29465

def small_package : ℕ := 6
def medium_package : ℕ := 15
def large_package : ℕ := 30

def louis_small_packages : ℕ := 5
def louis_medium_packages : ℕ := 3
def louis_large_packages : ℕ := 2

def louis_eaten : ℕ := 54
def num_friends : ℕ := 4

theorem lemon_heads_distribution :
  let total := louis_small_packages * small_package + 
               louis_medium_packages * medium_package + 
               louis_large_packages * large_package
  let remaining := total - louis_eaten
  let per_friend := remaining / num_friends
  per_friend = 3 * small_package + 2 ∧ 
  remaining % num_friends = 1 := by sorry

end NUMINAMATH_CALUDE_lemon_heads_distribution_l294_29465


namespace NUMINAMATH_CALUDE_ellipse_y_axis_l294_29448

/-- The equation represents an ellipse with focal points on the y-axis -/
theorem ellipse_y_axis (x y : ℝ) : 
  (x^2 / (Real.sin (Real.sqrt 2) - Real.sin (Real.sqrt 3))) + 
  (y^2 / (Real.cos (Real.sqrt 2) - Real.cos (Real.sqrt 3))) = 1 →
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_l294_29448


namespace NUMINAMATH_CALUDE_tank_capacity_l294_29470

theorem tank_capacity (bucket_capacity : ℚ) : 
  (13 * 42 = 91 * bucket_capacity) → bucket_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l294_29470


namespace NUMINAMATH_CALUDE_equation_solution_l294_29477

theorem equation_solution : 
  ∃ x : ℝ, (3*x - 5) / (x^2 - 7*x + 12) + (5*x - 1) / (x^2 - 5*x + 6) = (8*x - 13) / (x^2 - 6*x + 8) ∧ x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l294_29477


namespace NUMINAMATH_CALUDE_square_perimeter_l294_29445

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 324 → 
  area = side * side →
  perimeter = 4 * side →
  perimeter = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l294_29445


namespace NUMINAMATH_CALUDE_skateboard_distance_l294_29431

/-- The sum of an arithmetic sequence with first term 8, common difference 10, and 40 terms -/
theorem skateboard_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) : 
  a₁ = 8 → d = 10 → n = 40 → 
  (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = 8120 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l294_29431


namespace NUMINAMATH_CALUDE_exists_parallel_line_l294_29497

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (intersects : Plane → Plane → Prop)
variable (not_perpendicular : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem exists_parallel_line
  (α β γ : Plane)
  (h1 : perpendicular β γ)
  (h2 : intersects α γ)
  (h3 : not_perpendicular α γ) :
  ∃ (a : Line), in_plane a α ∧ parallel a γ :=
sorry

end NUMINAMATH_CALUDE_exists_parallel_line_l294_29497


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l294_29493

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
    (∀ (a' b' c' : ℕ), 
      (a' > 0 ∧ b' > 0 ∧ c' > 0) →
      (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c') →
      c ≤ c') ∧
    a + b + c = 113 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l294_29493


namespace NUMINAMATH_CALUDE_recycling_team_points_l294_29468

/-- Represents the recycling data for a team member -/
structure RecyclingData where
  paper : Nat
  plastic : Nat
  aluminum : Nat

/-- Calculates the points earned for a given recycling data -/
def calculate_points (data : RecyclingData) : Nat :=
  (data.paper / 12) + (data.plastic / 6) + (data.aluminum / 4)

/-- The recycling data for each team member -/
def team_data : List RecyclingData := [
  { paper := 35, plastic := 15, aluminum := 5 },   -- Zoe
  { paper := 28, plastic := 18, aluminum := 8 },   -- Friend 1
  { paper := 22, plastic := 10, aluminum := 6 },   -- Friend 2
  { paper := 40, plastic := 20, aluminum := 10 },  -- Friend 3
  { paper := 18, plastic := 12, aluminum := 8 }    -- Friend 4
]

/-- Theorem: The recycling team earned 28 points -/
theorem recycling_team_points : 
  (team_data.map calculate_points).sum = 28 := by
  sorry

end NUMINAMATH_CALUDE_recycling_team_points_l294_29468


namespace NUMINAMATH_CALUDE_multiplicative_inverse_300_mod_2399_l294_29451

theorem multiplicative_inverse_300_mod_2399 :
  (39 : ℤ)^2 + 80^2 = 89^2 →
  (300 * 1832) % 2399 = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_300_mod_2399_l294_29451


namespace NUMINAMATH_CALUDE_black_grid_probability_l294_29434

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Rotates the grid 90 degrees clockwise --/
def rotate (g : Grid) : Grid := sorry

/-- Applies the painting rule: white squares adjacent to black become black --/
def applyPaintRule (g : Grid) : Grid := sorry

/-- Checks if the entire grid is black --/
def allBlack (g : Grid) : Prop := ∀ i j, g i j = true

/-- Generates a random initial grid --/
def randomGrid : Grid := sorry

/-- The probability of a grid being entirely black after operations --/
def blackProbability : ℝ := sorry

theorem black_grid_probability :
  ∃ (p : ℝ), 0 < p ∧ p < 1 ∧ blackProbability = p := by sorry

end NUMINAMATH_CALUDE_black_grid_probability_l294_29434


namespace NUMINAMATH_CALUDE_largest_circle_equation_l294_29474

/-- The line equation ax - y - 4a - 2 = 0, where a is a real number -/
def line_equation (a x y : ℝ) : Prop := a * x - y - 4 * a - 2 = 0

/-- The center of the circle is at point (2, 0) -/
def circle_center : ℝ × ℝ := (2, 0)

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

theorem largest_circle_equation :
  ∃ (r : ℝ), r > 0 ∧
  (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y circle_center.1 circle_center.2 r) ∧
  (∀ r' : ℝ, r' > 0 →
    (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y circle_center.1 circle_center.2 r') →
    r' ≤ r) ∧
  (∀ x y : ℝ, circle_equation x y circle_center.1 circle_center.2 r ↔ (x - 2)^2 + y^2 = 8) :=
sorry

end NUMINAMATH_CALUDE_largest_circle_equation_l294_29474


namespace NUMINAMATH_CALUDE_volumeAsFractionOfLitre_l294_29419

-- Define the conversion factor from litres to millilitres
def litreToMl : ℝ := 1000

-- Define the volume in millilitres
def volumeMl : ℝ := 30

-- Theorem to prove
theorem volumeAsFractionOfLitre : (volumeMl / litreToMl) = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_volumeAsFractionOfLitre_l294_29419


namespace NUMINAMATH_CALUDE_quadratic_sequence_bound_l294_29446

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the solutions of a quadratic equation -/
structure QuadraticSolution where
  x₁ : ℝ
  x₂ : ℝ

/-- Function to get the next quadratic equation in the sequence -/
def nextEquation (eq : QuadraticEquation) (sol : QuadraticSolution) : QuadraticEquation :=
  { a := 1, b := -sol.x₁, c := -sol.x₂ }

/-- Theorem stating that the sequence of quadratic equations has at most 5 elements -/
theorem quadratic_sequence_bound
  (a₁ b₁ : ℝ)
  (h₁ : a₁ ≠ 0)
  (h₂ : b₁ ≠ 0)
  (initial : QuadraticEquation)
  (h₃ : initial = { a := 1, b := a₁, c := b₁ })
  (next : QuadraticEquation → QuadraticSolution → QuadraticEquation)
  (h₄ : ∀ eq sol, next eq sol = nextEquation eq sol) :
  ∃ n : ℕ, n ≤ 5 ∧ ∀ m : ℕ, m > n →
    ¬∃ (seq : ℕ → QuadraticEquation) (sols : ℕ → QuadraticSolution),
      (seq 0 = initial) ∧
      (∀ k < m, seq (k + 1) = next (seq k) (sols k)) ∧
      (∀ k < m, (sols k).x₁ ≤ (sols k).x₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_bound_l294_29446


namespace NUMINAMATH_CALUDE_magnitude_of_c_l294_29430

def vector_a : Fin 2 → ℝ := ![1, -1]
def vector_b : Fin 2 → ℝ := ![2, 1]

def vector_c : Fin 2 → ℝ := λ i => 2 * vector_a i + vector_b i

theorem magnitude_of_c :
  Real.sqrt ((vector_c 0) ^ 2 + (vector_c 1) ^ 2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_c_l294_29430


namespace NUMINAMATH_CALUDE_single_elimination_tournament_matches_l294_29424

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  initial_players : ℕ
  matches_played : ℕ

/-- The number of matches needed to determine a champion in a single-elimination tournament -/
def matches_needed (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

theorem single_elimination_tournament_matches 
  (tournament : SingleEliminationTournament)
  (h : tournament.initial_players = 512) :
  matches_needed tournament = 511 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_matches_l294_29424


namespace NUMINAMATH_CALUDE_larger_number_problem_l294_29469

theorem larger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 147)
  (relation : x = 0.375 * y + 4)
  (x_larger : x > y) : 
  x = 43 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l294_29469


namespace NUMINAMATH_CALUDE_power_function_problem_l294_29420

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- Define the problem statement
theorem power_function_problem (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 3 = Real.sqrt 3) : 
  f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_problem_l294_29420


namespace NUMINAMATH_CALUDE_total_salary_is_7600_l294_29488

/-- Represents the weekly working hours for each employee -/
structure WeeklyHours where
  fiona : ℕ
  john : ℕ
  jeremy : ℕ

/-- Represents the hourly wage -/
def hourlyWage : ℚ := 20

/-- Represents the number of weeks in a month -/
def weeksPerMonth : ℕ := 4

/-- Calculates the monthly salary for an employee -/
def monthlySalary (hours : ℕ) : ℚ :=
  hours * hourlyWage * weeksPerMonth

/-- Calculates the total monthly expenditure on salaries -/
def totalMonthlyExpenditure (hours : WeeklyHours) : ℚ :=
  monthlySalary hours.fiona + monthlySalary hours.john + monthlySalary hours.jeremy

/-- Theorem stating that the total monthly expenditure on salaries is $7600 -/
theorem total_salary_is_7600 (hours : WeeklyHours)
    (h1 : hours.fiona = 40)
    (h2 : hours.john = 30)
    (h3 : hours.jeremy = 25) :
    totalMonthlyExpenditure hours = 7600 := by
  sorry

end NUMINAMATH_CALUDE_total_salary_is_7600_l294_29488


namespace NUMINAMATH_CALUDE_king_total_payment_l294_29480

def crown_cost : ℚ := 20000
def architect_cost : ℚ := 50000
def chef_cost : ℚ := 10000

def crown_tip_percent : ℚ := 10 / 100
def architect_tip_percent : ℚ := 5 / 100
def chef_tip_percent : ℚ := 15 / 100

def total_cost : ℚ := crown_cost * (1 + crown_tip_percent) + 
                       architect_cost * (1 + architect_tip_percent) + 
                       chef_cost * (1 + chef_tip_percent)

theorem king_total_payment : total_cost = 86000 :=
by sorry

end NUMINAMATH_CALUDE_king_total_payment_l294_29480


namespace NUMINAMATH_CALUDE_circle_plus_four_two_l294_29417

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 2 * a + 5 * b

-- Statement to prove
theorem circle_plus_four_two : circle_plus 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_four_two_l294_29417


namespace NUMINAMATH_CALUDE_trevors_age_l294_29491

theorem trevors_age (T : ℕ) : 
  (20 + (24 - T) = 3 * T) → T = 11 := by
  sorry

end NUMINAMATH_CALUDE_trevors_age_l294_29491


namespace NUMINAMATH_CALUDE_brand_a_households_l294_29412

theorem brand_a_households (total : ℕ) (neither : ℕ) (both : ℕ) (ratio : ℕ) :
  total = 160 →
  neither = 80 →
  both = 5 →
  ratio = 3 →
  ∃ (only_a only_b : ℕ),
    total = neither + only_a + only_b + both ∧
    only_b = ratio * both ∧
    only_a = 60 :=
by sorry

end NUMINAMATH_CALUDE_brand_a_households_l294_29412


namespace NUMINAMATH_CALUDE_critical_point_theorem_l294_29458

def sequence_property (x : ℕ → ℝ) : Prop :=
  (∀ n, x n > 0) ∧
  (8 * x 2 - 7 * x 1) * (x 1)^7 = 8 ∧
  (∀ k ≥ 2, (x (k+1)) * (x (k-1)) - (x k)^2 = ((x (k-1))^8 - (x k)^8) / ((x k)^7 * (x (k-1))^7))

def monotonically_decreasing (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n+1) ≤ x n

def not_monotonic (x : ℕ → ℝ) : Prop :=
  ∃ m n, m < n ∧ x m < x n

theorem critical_point_theorem (x : ℕ → ℝ) (h : sequence_property x) :
  ∃ a : ℝ, a = 8^(1/8) ∧
    ((x 1 > a → monotonically_decreasing x) ∧
     (0 < x 1 ∧ x 1 < a → not_monotonic x)) :=
sorry

end NUMINAMATH_CALUDE_critical_point_theorem_l294_29458


namespace NUMINAMATH_CALUDE_stating_largest_cone_in_cube_l294_29439

/-- Represents the dimensions of a cone carved from a cube. -/
structure ConeDimensions where
  height : ℝ
  baseRadius : ℝ
  volume : ℝ

/-- 
Theorem stating the dimensions of the largest cone that can be carved from a cube.
The cone's axis coincides with one of the cube's body diagonals.
-/
theorem largest_cone_in_cube (a : ℝ) (ha : a > 0) : 
  ∃ (cone : ConeDimensions), 
    cone.height = a * Real.sqrt 3 / 2 ∧
    cone.baseRadius = a * Real.sqrt 3 / (2 * Real.sqrt 2) ∧
    cone.volume = π * a^3 * Real.sqrt 3 / 16 ∧
    ∀ (other : ConeDimensions), other.volume ≤ cone.volume := by
  sorry

end NUMINAMATH_CALUDE_stating_largest_cone_in_cube_l294_29439


namespace NUMINAMATH_CALUDE_intersection_points_l294_29475

-- Define g as a function from real numbers to real numbers
variable (g : ℝ → ℝ)

-- Define the property that g is invertible
def IsInvertible (g : ℝ → ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x, h (g x) = x) ∧ (∀ y, g (h y) = y)

-- Theorem statement
theorem intersection_points (h : IsInvertible g) :
  (∃! n : Nat, ∃ s : Finset ℝ, s.card = n ∧
    (∀ x : ℝ, x ∈ s ↔ g (x^3) = g (x^6))) ∧
  (∃ s : Finset ℝ, s.card = 3 ∧
    (∀ x : ℝ, x ∈ s ↔ g (x^3) = g (x^6))) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_l294_29475


namespace NUMINAMATH_CALUDE_unique_solution_is_one_l294_29409

theorem unique_solution_is_one (n : ℕ) (hn : n ≥ 1) :
  (∃ (a b : ℕ), 
    (∀ (p : ℕ), Prime p → ¬(p^3 ∣ (a^2 + b + 3))) ∧
    ((a * b + 3 * b + 8) : ℚ) / (a^2 + b + 3 : ℚ) = n) 
  ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_one_l294_29409


namespace NUMINAMATH_CALUDE_f_max_value_l294_29403

/-- The quadratic function f(x) = -x^2 + 2x + 4 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

/-- The maximum value of f(x) is 5 -/
theorem f_max_value : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
  sorry

end NUMINAMATH_CALUDE_f_max_value_l294_29403


namespace NUMINAMATH_CALUDE_triangle_sides_not_proportional_l294_29400

theorem triangle_sides_not_proportional (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ¬∃ (m : ℝ), m > 0 ∧ a = m^a ∧ b = m^b ∧ c = m^c :=
sorry

end NUMINAMATH_CALUDE_triangle_sides_not_proportional_l294_29400


namespace NUMINAMATH_CALUDE_clarinet_fraction_in_band_l294_29461

theorem clarinet_fraction_in_band (total_band : ℕ) (flutes_in : ℕ) (trumpets_in : ℕ) (pianists_in : ℕ) (clarinets_total : ℕ) :
  total_band = 53 →
  flutes_in = 16 →
  trumpets_in = 20 →
  pianists_in = 2 →
  clarinets_total = 30 →
  (total_band - (flutes_in + trumpets_in + pianists_in)) / clarinets_total = 1 / 2 :=
by
  sorry

#check clarinet_fraction_in_band

end NUMINAMATH_CALUDE_clarinet_fraction_in_band_l294_29461


namespace NUMINAMATH_CALUDE_least_sum_exponents_3125_l294_29432

theorem least_sum_exponents_3125 : 
  let n := 3125
  let is_valid_representation (rep : List ℕ) := 
    (rep.map (λ i => 2^i)).sum = n ∧ rep.Nodup
  ∃ (rep : List ℕ), is_valid_representation rep ∧
    ∀ (other_rep : List ℕ), is_valid_representation other_rep → 
      rep.sum ≤ other_rep.sum :=
by sorry

end NUMINAMATH_CALUDE_least_sum_exponents_3125_l294_29432


namespace NUMINAMATH_CALUDE_guitar_difference_is_three_l294_29453

/-- The number of fewer 8 string guitars compared to normal guitars -/
def guitar_difference : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_normal_guitars : ℕ := 2 * num_basses
  let strings_per_normal_guitar : ℕ := 6
  let strings_per_8string_guitar : ℕ := 8
  let total_strings : ℕ := 72
  let normal_guitar_strings : ℕ := num_normal_guitars * strings_per_normal_guitar
  let bass_strings : ℕ := num_basses * strings_per_bass
  let remaining_strings : ℕ := total_strings - (normal_guitar_strings + bass_strings)
  let num_8string_guitars : ℕ := remaining_strings / strings_per_8string_guitar
  num_normal_guitars - num_8string_guitars

theorem guitar_difference_is_three :
  guitar_difference = 3 := by sorry

end NUMINAMATH_CALUDE_guitar_difference_is_three_l294_29453


namespace NUMINAMATH_CALUDE_car_trip_average_mpg_l294_29498

/-- Proves that the average miles per gallon for a car trip is 450/11, given specific conditions. -/
theorem car_trip_average_mpg :
  -- Define the distance from B to C as x
  ∀ x : ℝ,
  x > 0 →
  -- Distance from A to B is twice the distance from B to C
  let dist_ab := 2 * x
  let dist_bc := x
  -- Define the fuel efficiencies
  let mpg_ab := 25
  let mpg_bc := 30
  -- Calculate total distance and total fuel used
  let total_dist := dist_ab + dist_bc
  let total_fuel := dist_ab / mpg_ab + dist_bc / mpg_bc
  -- The average MPG for the entire trip
  let avg_mpg := total_dist / total_fuel
  -- Prove that the average MPG equals 450/11
  avg_mpg = 450 / 11 := by
    sorry

#eval (450 : ℚ) / 11

end NUMINAMATH_CALUDE_car_trip_average_mpg_l294_29498


namespace NUMINAMATH_CALUDE_sum_of_digits_is_400_l294_29437

/-- A number system with base r -/
structure BaseR where
  r : ℕ
  h_r : r ≤ 400

/-- A number x in base r of the form ppqq -/
structure NumberX (b : BaseR) where
  p : ℕ
  q : ℕ
  h_pq : 7 * q = 17 * p
  x : ℕ
  h_x : x = p * b.r^3 + p * b.r^2 + q * b.r + q

/-- The square of x is a seven-digit palindrome with middle digit zero -/
def is_palindrome_square (b : BaseR) (x : NumberX b) : Prop :=
  ∃ (a c : ℕ),
    x.x^2 = a * b.r^6 + c * b.r^5 + c * b.r^4 + 0 * b.r^3 + c * b.r^2 + c * b.r + a

/-- The sum of digits of x^2 in base r -/
def sum_of_digits (b : BaseR) (x : NumberX b) : ℕ :=
  sorry  -- Definition of sum of digits

/-- Main theorem -/
theorem sum_of_digits_is_400 (b : BaseR) (x : NumberX b) 
    (h_palindrome : is_palindrome_square b x) : 
    sum_of_digits b x = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_400_l294_29437


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_and_house_l294_29466

theorem blocks_used_for_tower_and_house : 
  let total_blocks : ℕ := 58
  let tower_blocks : ℕ := 27
  let house_blocks : ℕ := 53
  tower_blocks + house_blocks = 80 :=
by sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_and_house_l294_29466


namespace NUMINAMATH_CALUDE_second_expression_value_l294_29423

/-- Given that the average of (2a + 16) and x is 79, and a = 30, prove that x = 82 -/
theorem second_expression_value (a x : ℝ) : 
  ((2 * a + 16) + x) / 2 = 79 → a = 30 → x = 82 := by
  sorry

end NUMINAMATH_CALUDE_second_expression_value_l294_29423


namespace NUMINAMATH_CALUDE_geese_count_l294_29479

/-- Given a marsh with ducks and geese, calculate the number of geese -/
theorem geese_count (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_count_l294_29479


namespace NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l294_29435

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + a)^3

-- State the theorem
theorem sum_f_two_and_neg_two (a : ℝ) : 
  (∀ x : ℝ, f a (1 + x) = -f a (1 - x)) → f a 2 + f a (-2) = -26 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l294_29435


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l294_29492

/-- Given real numbers a, b, and c, and polynomials g and f as defined,
    prove that f(2) = 40640 -/
theorem polynomial_root_relation (a b c : ℝ) : 
  let g := fun (x : ℝ) => x^3 + a*x^2 + x + 20
  let f := fun (x : ℝ) => x^4 + x^3 + b*x^2 + 50*x + c
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g x = 0 ∧ g y = 0 ∧ g z = 0) →
  (∀ (x : ℝ), g x = 0 → f x = 0) →
  f 2 = 40640 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l294_29492


namespace NUMINAMATH_CALUDE_total_shaded_area_specific_total_shaded_area_l294_29401

/-- The total shaded area of three overlapping rectangles -/
theorem total_shaded_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ)
  (shared_side triple_overlap_width : ℕ) : ℕ :=
  let rect1_area := rect1_width * rect1_height
  let rect2_area := rect2_width * rect2_height
  let rect3_area := rect3_width * rect3_height
  let overlap_area := shared_side * shared_side
  let triple_overlap_area := triple_overlap_width * shared_side
  rect1_area + rect2_area + rect3_area - overlap_area - triple_overlap_area

/-- The total shaded area of the specific configuration is 136 square units -/
theorem specific_total_shaded_area :
  total_shaded_area 4 15 5 10 3 18 4 3 = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_specific_total_shaded_area_l294_29401


namespace NUMINAMATH_CALUDE_inequality_theorem_l294_29476

theorem inequality_theorem (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : 1/p + 1/q^2 = 1) : 
  1/(p*(p+2)) + 1/(q*(q+2)) ≥ (21*Real.sqrt 21 - 71)/80 ∧
  (1/(p*(p+2)) + 1/(q*(q+2)) = (21*Real.sqrt 21 - 71)/80 ↔ 
    p = 2 + 2*Real.sqrt (7/3) ∧ q = (Real.sqrt 21 + 1)/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l294_29476


namespace NUMINAMATH_CALUDE_angle_measure_l294_29494

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l294_29494


namespace NUMINAMATH_CALUDE_molly_total_distance_l294_29416

/-- The total distance Molly swam over two days -/
def total_distance (saturday_distance sunday_distance : ℕ) : ℕ :=
  saturday_distance + sunday_distance

/-- Theorem stating that Molly's total swimming distance is 430 meters -/
theorem molly_total_distance : total_distance 250 180 = 430 := by
  sorry

end NUMINAMATH_CALUDE_molly_total_distance_l294_29416


namespace NUMINAMATH_CALUDE_simplify_sum_of_radicals_l294_29462

theorem simplify_sum_of_radicals : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_sum_of_radicals_l294_29462


namespace NUMINAMATH_CALUDE_symmetric_complex_sum_l294_29415

theorem symmetric_complex_sum (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  let w : ℂ := Complex.I * (Complex.I - 2)
  (z.re = w.re ∧ z.im = -w.im) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_sum_l294_29415


namespace NUMINAMATH_CALUDE_smallest_N_eight_works_smallest_N_is_8_l294_29440

theorem smallest_N : ∀ N : ℕ+, 
  (∃ a b c d : ℕ, 
    a = N.val * 125 / 1000 ∧
    b = N.val * 500 / 1000 ∧
    c = N.val * 250 / 1000 ∧
    d = N.val * 125 / 1000) →
  N.val ≥ 8 :=
by sorry

theorem eight_works : 
  ∃ a b c d : ℕ,
    a = 8 * 125 / 1000 ∧
    b = 8 * 500 / 1000 ∧
    c = 8 * 250 / 1000 ∧
    d = 8 * 125 / 1000 :=
by sorry

theorem smallest_N_is_8 : 
  ∀ N : ℕ+, 
    (∃ a b c d : ℕ, 
      a = N.val * 125 / 1000 ∧
      b = N.val * 500 / 1000 ∧
      c = N.val * 250 / 1000 ∧
      d = N.val * 125 / 1000) ↔
    N.val ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_eight_works_smallest_N_is_8_l294_29440


namespace NUMINAMATH_CALUDE_power_equality_l294_29426

theorem power_equality (m : ℕ) : 5^m = 5 * 25^5 * 125^3 → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l294_29426


namespace NUMINAMATH_CALUDE_continuity_at_two_l294_29484

noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x^2 - 4)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_two_l294_29484


namespace NUMINAMATH_CALUDE_range_of_x_minus_sqrt3y_l294_29486

theorem range_of_x_minus_sqrt3y (x y : ℝ) 
  (h : x^2 + y^2 - 2*x + 2*Real.sqrt 3*y + 3 = 0) :
  ∃ (min max : ℝ), min = 2 ∧ max = 6 ∧ 
    (∀ z, z = x - Real.sqrt 3 * y → min ≤ z ∧ z ≤ max) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_minus_sqrt3y_l294_29486


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l294_29407

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem: It is false that all squares are congruent to each other
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (for completeness, not used in the proof)
def convex (s : Square) : Prop := true
def equiangular (s : Square) : Prop := true
def regular_polygon (s : Square) : Prop := true
def similar (s1 s2 : Square) : Prop := true

end NUMINAMATH_CALUDE_not_all_squares_congruent_l294_29407


namespace NUMINAMATH_CALUDE_length_of_AB_l294_29402

-- Define the triangle
def Triangle (A B C : ℝ) := True

-- Define the right angle
def is_right_angle (B : ℝ) := B = 90

-- Define the angle A
def angle_A (A : ℝ) := A = 40

-- Define the length of side BC
def side_BC (BC : ℝ) := BC = 7

-- Theorem statement
theorem length_of_AB (A B C BC : ℝ) 
  (triangle : Triangle A B C) 
  (right_angle : is_right_angle B) 
  (angle_a : angle_A A) 
  (side_bc : side_BC BC) : 
  ∃ (AB : ℝ), abs (AB - 8.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_l294_29402


namespace NUMINAMATH_CALUDE_count_integers_in_range_l294_29456

theorem count_integers_in_range : 
  (Finset.range 1001).card = (Finset.Ico 1000 2001).card := by sorry

end NUMINAMATH_CALUDE_count_integers_in_range_l294_29456


namespace NUMINAMATH_CALUDE_tank_fill_time_l294_29490

/-- Represents a pipe with a flow rate (positive for filling, negative for draining) -/
structure Pipe where
  rate : Int

/-- Represents a tank with a capacity and a list of pipes -/
structure Tank where
  capacity : Nat
  pipes : List Pipe

def cycleTime : Nat := 3

def cycleVolume (tank : Tank) : Int :=
  tank.pipes.foldl (fun acc pipe => acc + pipe.rate) 0

theorem tank_fill_time (tank : Tank) (h1 : tank.capacity = 750)
    (h2 : tank.pipes = [⟨40⟩, ⟨30⟩, ⟨-20⟩])
    (h3 : cycleVolume tank = 50)
    (h4 : tank.capacity / cycleVolume tank * cycleTime = 45) :
  ∃ (t : Nat), t = 45 ∧ t * cycleVolume tank ≥ tank.capacity := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l294_29490


namespace NUMINAMATH_CALUDE_intersection_M_N_l294_29496

def M : Set ℝ := {x | (x + 2) * (x - 2) > 0}
def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_M_N : M ∩ N = {-3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l294_29496


namespace NUMINAMATH_CALUDE_longest_wait_time_l294_29495

def initial_wait : ℕ := 20

def license_renewal_wait (t : ℕ) : ℕ := 2 * t + 8

def registration_update_wait (t : ℕ) : ℕ := 4 * t + 14

def driving_record_wait (t : ℕ) : ℕ := 3 * t - 16

theorem longest_wait_time :
  let tasks := [initial_wait,
                license_renewal_wait initial_wait,
                registration_update_wait initial_wait,
                driving_record_wait initial_wait]
  registration_update_wait initial_wait = 94 ∧
  ∀ t ∈ tasks, t ≤ registration_update_wait initial_wait :=
by sorry

end NUMINAMATH_CALUDE_longest_wait_time_l294_29495


namespace NUMINAMATH_CALUDE_share_ratio_problem_l294_29422

theorem share_ratio_problem (total : ℝ) (share_A : ℝ) (ratio_B_C : ℚ) 
  (h_total : total = 116000)
  (h_share_A : share_A = 29491.525423728814)
  (h_ratio_B_C : ratio_B_C = 5/6) :
  ∃ (share_B : ℝ), share_A / share_B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_problem_l294_29422


namespace NUMINAMATH_CALUDE_total_flight_distance_l294_29452

/-- The total distance to fly from Germany to Russia and then return to Spain,
    given the distances between Spain-Russia and Spain-Germany. -/
theorem total_flight_distance (spain_russia spain_germany : ℕ) 
  (h1 : spain_russia = 7019)
  (h2 : spain_germany = 1615) :
  spain_russia + (spain_russia - spain_germany) = 12423 :=
by sorry

end NUMINAMATH_CALUDE_total_flight_distance_l294_29452


namespace NUMINAMATH_CALUDE_no_constant_absolute_value_inequality_l294_29418

theorem no_constant_absolute_value_inequality :
  ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ), 
    |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| := by
  sorry

end NUMINAMATH_CALUDE_no_constant_absolute_value_inequality_l294_29418


namespace NUMINAMATH_CALUDE_runners_meeting_time_l294_29454

def anna_lap_time : ℕ := 5
def bob_lap_time : ℕ := 8
def carol_lap_time : ℕ := 10

def meeting_time : ℕ := 40

theorem runners_meeting_time :
  Nat.lcm (Nat.lcm anna_lap_time bob_lap_time) carol_lap_time = meeting_time :=
sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l294_29454


namespace NUMINAMATH_CALUDE_flower_expense_proof_l294_29482

/-- Calculates the total expense for flowers given the quantities and price per flower -/
def totalExpense (tulips carnations roses : ℕ) (pricePerFlower : ℕ) : ℕ :=
  (tulips + carnations + roses) * pricePerFlower

/-- Proves that the total expense for the given flower quantities and price is 1890 -/
theorem flower_expense_proof :
  totalExpense 250 375 320 2 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_flower_expense_proof_l294_29482


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l294_29427

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l294_29427
