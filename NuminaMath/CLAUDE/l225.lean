import Mathlib

namespace NUMINAMATH_CALUDE_next_number_is_1461_l225_22555

/-- Represents the sequence generator function -/
def sequenceGenerator (n : ℕ) : ℕ := 
  100 + 15 + (n * (n + 1))

/-- Proves that the next number after 1445 in the sequence is 1461 -/
theorem next_number_is_1461 : 
  ∃ k, sequenceGenerator k = 1445 ∧ sequenceGenerator (k + 1) = 1461 :=
sorry

end NUMINAMATH_CALUDE_next_number_is_1461_l225_22555


namespace NUMINAMATH_CALUDE_haley_money_difference_l225_22582

/-- Calculates the difference between the final and initial amount of money Haley has after various transactions. -/
theorem haley_money_difference :
  let initial_amount : ℚ := 2
  let chores_earnings : ℚ := 5.25
  let birthday_gift : ℚ := 10
  let neighbor_help : ℚ := 7.5
  let found_money : ℚ := 0.5
  let aunt_gift_pounds : ℚ := 3
  let pound_to_dollar : ℚ := 1.3
  let candy_spent : ℚ := 3.75
  let money_lost : ℚ := 1.5
  
  let total_received : ℚ := chores_earnings + birthday_gift + neighbor_help + found_money + aunt_gift_pounds * pound_to_dollar
  let total_spent : ℚ := candy_spent + money_lost
  let final_amount : ℚ := initial_amount + total_received - total_spent
  
  final_amount - initial_amount = 19.9 := by sorry

end NUMINAMATH_CALUDE_haley_money_difference_l225_22582


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l225_22597

/-- The area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 20) 
  (h_inradius : inradius = 3) : 
  area = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l225_22597


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l225_22547

theorem hot_dogs_remainder :
  25197625 % 4 = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l225_22547


namespace NUMINAMATH_CALUDE_koolaid_water_increase_factor_l225_22550

/-- Proves that the water increase factor is 4 given the initial conditions and final percentage --/
theorem koolaid_water_increase_factor : 
  ∀ (initial_koolaid initial_water evaporated_water : ℚ)
    (final_percentage : ℚ),
  initial_koolaid = 2 →
  initial_water = 16 →
  evaporated_water = 4 →
  final_percentage = 4/100 →
  ∃ (increase_factor : ℚ),
    increase_factor = 4 ∧
    initial_koolaid / (initial_koolaid + (initial_water - evaporated_water) * increase_factor) = final_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_koolaid_water_increase_factor_l225_22550


namespace NUMINAMATH_CALUDE_binomial_max_term_max_term_sqrt_seven_l225_22527

theorem binomial_max_term (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
sorry

theorem max_term_sqrt_seven :
  let n := 205
  let x := Real.sqrt 7
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j ∧
    k = 149 :=
sorry

end NUMINAMATH_CALUDE_binomial_max_term_max_term_sqrt_seven_l225_22527


namespace NUMINAMATH_CALUDE_initial_total_marbles_l225_22522

/-- Represents the number of parts in the ratio for each person -/
def brittany_ratio : ℕ := 3
def alex_ratio : ℕ := 5
def jamy_ratio : ℕ := 7

/-- Represents the total number of marbles Alex has after receiving half of Brittany's marbles -/
def alex_final_marbles : ℕ := 260

/-- The theorem stating the initial total number of marbles -/
theorem initial_total_marbles :
  ∃ (x : ℕ),
    (brittany_ratio * x + alex_ratio * x + jamy_ratio * x = 600) ∧
    (alex_ratio * x + (brittany_ratio * x) / 2 = alex_final_marbles) :=
by sorry

end NUMINAMATH_CALUDE_initial_total_marbles_l225_22522


namespace NUMINAMATH_CALUDE_equation_solution_l225_22584

theorem equation_solution : 
  ∃ x : ℝ, (x / (x - 1) - 1 = 1) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l225_22584


namespace NUMINAMATH_CALUDE_tree_height_difference_l225_22594

theorem tree_height_difference :
  let pine_height : ℚ := 53/4
  let maple_height : ℚ := 41/2
  maple_height - pine_height = 29/4 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l225_22594


namespace NUMINAMATH_CALUDE_range_of_g_l225_22513

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := f x - 2*x

-- Define the interval
def I : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_g :
  {y | ∃ x ∈ I, g x = y} = {y | -1 ≤ y ∧ y ≤ 8} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l225_22513


namespace NUMINAMATH_CALUDE_three_zeros_l225_22568

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2*a + 2) * Real.log x + b

theorem three_zeros (a b : ℝ) (ha : a > 3) (hb : a^2 + a + 1 < b) (hb' : b < 2*a^2 - 2*a + 2) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f a b x = 0 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_l225_22568


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l225_22585

/-- Given 6 people in an elevator with an average weight of 152 lbs, 
    prove that if a 7th person enters and the new average becomes 151 lbs, 
    then the weight of the 7th person is 145 lbs. -/
theorem elevator_weight_problem (people : ℕ) (avg_weight_before : ℝ) (avg_weight_after : ℝ) :
  people = 6 →
  avg_weight_before = 152 →
  avg_weight_after = 151 →
  (people * avg_weight_before + (avg_weight_after * (people + 1) - people * avg_weight_before)) = 145 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l225_22585


namespace NUMINAMATH_CALUDE_base_conversion_l225_22576

theorem base_conversion (b : ℝ) (h : b > 0) : 53 = 1 * b^2 + 0 * b + 3 → b = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l225_22576


namespace NUMINAMATH_CALUDE_system_solution_existence_l225_22588

theorem system_solution_existence (b : ℝ) : 
  (∃ (a x y : ℝ), y = -b - x^2 ∧ x^2 + y^2 + 8*a^2 = 4 + 4*a*(x + y)) ↔ 
  b ≤ 2 * Real.sqrt 2 + 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l225_22588


namespace NUMINAMATH_CALUDE_next_coincidence_l225_22540

def factory_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def town_hall_interval : ℕ := 30

theorem next_coincidence (start_time : ℕ) :
  ∃ (t : ℕ), t > start_time ∧ 
  t % factory_interval = 0 ∧
  t % fire_station_interval = 0 ∧
  t % town_hall_interval = 0 ∧
  t - start_time = 360 := by
sorry

end NUMINAMATH_CALUDE_next_coincidence_l225_22540


namespace NUMINAMATH_CALUDE_dot_product_OA_OB_is_zero_l225_22563

theorem dot_product_OA_OB_is_zero (OA OB : ℝ × ℝ) : 
  OA = (1, -3) →
  ‖OA‖ = ‖OB‖ →
  ‖OA - OB‖ = 2 * Real.sqrt 5 →
  OA • OB = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_OA_OB_is_zero_l225_22563


namespace NUMINAMATH_CALUDE_expression_evaluation_l225_22536

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l225_22536


namespace NUMINAMATH_CALUDE_managers_in_game_l225_22574

/-- The number of managers participating in a volleyball game --/
def num_managers (total_teams : ℕ) (people_per_team : ℕ) (num_employees : ℕ) : ℕ :=
  total_teams * people_per_team - num_employees

/-- Theorem stating that the number of managers in the game is 3 --/
theorem managers_in_game :
  num_managers 3 2 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_managers_in_game_l225_22574


namespace NUMINAMATH_CALUDE_bellas_score_l225_22577

theorem bellas_score (total_students : ℕ) (avg_without_bella : ℚ) (avg_with_bella : ℚ) :
  total_students = 20 →
  avg_without_bella = 82 →
  avg_with_bella = 85 →
  (total_students * avg_with_bella - (total_students - 1) * avg_without_bella : ℚ) = 142 :=
by sorry

end NUMINAMATH_CALUDE_bellas_score_l225_22577


namespace NUMINAMATH_CALUDE_not_p_and_not_q_implies_not_p_and_not_q_l225_22583

theorem not_p_and_not_q_implies_not_p_and_not_q (p q : Prop) :
  (¬p ∧ ¬q) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_implies_not_p_and_not_q_l225_22583


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l225_22532

def A : Set ℝ := {x | x - 1 < 5}
def B : Set ℝ := {x | -4*x + 8 < 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l225_22532


namespace NUMINAMATH_CALUDE_machine_quality_comparison_l225_22544

/-- Represents a machine producing products of different quality classes -/
structure Machine where
  first_class : ℕ
  second_class : ℕ

/-- Calculates the frequency of first-class products for a machine -/
def first_class_frequency (m : Machine) : ℚ :=
  m.first_class / (m.first_class + m.second_class)

/-- Calculates the K² statistic for comparing two machines -/
def k_squared (m1 m2 : Machine) : ℚ :=
  let n := m1.first_class + m1.second_class + m2.first_class + m2.second_class
  let a := m1.first_class
  let b := m1.second_class
  let c := m2.first_class
  let d := m2.second_class
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The theorem to be proved -/
theorem machine_quality_comparison (machine_a machine_b : Machine)
  (h_a : machine_a = ⟨150, 50⟩)
  (h_b : machine_b = ⟨120, 80⟩) :
  first_class_frequency machine_a = 3/4 ∧
  first_class_frequency machine_b = 3/5 ∧
  k_squared machine_a machine_b > 6635/1000 := by
  sorry

end NUMINAMATH_CALUDE_machine_quality_comparison_l225_22544


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_max_a_value_exists_x_for_a_eq_3_l225_22518

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the maximum value of a
theorem max_a_value (a : ℝ) :
  (∃ x : ℝ, f x ≤ -a^2 + a + 7) → a ≤ 3 := by sorry

-- Theorem that 3 is indeed the maximum value
theorem exists_x_for_a_eq_3 :
  ∃ x : ℝ, f x ≤ -3^2 + 3 + 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_max_a_value_exists_x_for_a_eq_3_l225_22518


namespace NUMINAMATH_CALUDE_roots_of_equation_l225_22528

def f (x : ℝ) : ℝ := (x^3 - 3*x^2 + 2*x)*(x - 5)

theorem roots_of_equation : 
  {x : ℝ | f x = 0} = {0, 1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l225_22528


namespace NUMINAMATH_CALUDE_roots_sum_l225_22593

theorem roots_sum (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = 0 ↔ x = m ∨ x = n) → 
  m = 2*n → 
  m + n = 3/2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_l225_22593


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l225_22520

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The problem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a)
    (h_a3 : a 3 = 6)
    (h_sum : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l225_22520


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l225_22514

theorem existence_of_counterexample (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ b, c * b^2 ≥ a * b^2 := by
sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l225_22514


namespace NUMINAMATH_CALUDE_cos_arcsin_half_l225_22526

theorem cos_arcsin_half : Real.cos (Real.arcsin (1/2)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_half_l225_22526


namespace NUMINAMATH_CALUDE_max_value_of_f_l225_22510

/-- The quadratic function f(x) = -3x^2 + 6x + 2 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 2

/-- The maximum value of f(x) is 5 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l225_22510


namespace NUMINAMATH_CALUDE_first_digit_of_87_base_5_l225_22505

def base_5_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

def first_digit_base_5 (n : ℕ) : ℕ :=
  match base_5_representation n with
  | [] => 0  -- This case should never occur for a valid input
  | d::_ => d

theorem first_digit_of_87_base_5 :
  first_digit_base_5 87 = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_of_87_base_5_l225_22505


namespace NUMINAMATH_CALUDE_wood_measurement_theorem_l225_22533

/-- Represents the measurement of a piece of wood with a rope -/
structure WoodMeasurement where
  wood_length : ℝ
  rope_length : ℝ
  surplus : ℝ
  half_rope_shortage : ℝ

/-- The system of equations accurately represents the wood measurement situation -/
def accurate_representation (m : WoodMeasurement) : Prop :=
  (m.rope_length = m.wood_length + m.surplus) ∧
  (0.5 * m.rope_length = m.wood_length - m.half_rope_shortage)

/-- Theorem stating that the given conditions lead to the correct system of equations -/
theorem wood_measurement_theorem (m : WoodMeasurement) 
  (h1 : m.surplus = 4.5)
  (h2 : m.half_rope_shortage = 1) :
  accurate_representation m := by
  sorry

end NUMINAMATH_CALUDE_wood_measurement_theorem_l225_22533


namespace NUMINAMATH_CALUDE_train_length_l225_22589

/-- The length of a train that crosses two platforms of different lengths in given times. -/
theorem train_length 
  (platform1_length : ℝ) 
  (platform1_time : ℝ) 
  (platform2_length : ℝ) 
  (platform2_time : ℝ) 
  (h1 : platform1_length = 170)
  (h2 : platform1_time = 15)
  (h3 : platform2_length = 250)
  (h4 : platform2_time = 20) :
  ∃ (train_length : ℝ), 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧ 
    train_length = 70 := by
sorry


end NUMINAMATH_CALUDE_train_length_l225_22589


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l225_22509

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 18 * Real.sqrt 8 = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l225_22509


namespace NUMINAMATH_CALUDE_distances_product_bound_l225_22567

/-- Given an equilateral triangle with side length 1 and a point P inside it,
    the distances from P to the three sides satisfy 0 < ab + bc + ca ≤ 1/4 -/
theorem distances_product_bound (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = Real.sqrt 3 / 2 → 
  0 < a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_distances_product_bound_l225_22567


namespace NUMINAMATH_CALUDE_jill_watch_time_l225_22529

/-- The total time Jill spent watching shows, given the length of the first show and a multiplier for the second show. -/
def total_watch_time (first_show_length : ℕ) (second_show_multiplier : ℕ) : ℕ :=
  first_show_length + first_show_length * second_show_multiplier

/-- Theorem stating that Jill spent 150 minutes watching shows. -/
theorem jill_watch_time : total_watch_time 30 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jill_watch_time_l225_22529


namespace NUMINAMATH_CALUDE_grinder_loss_percentage_l225_22579

/-- Represents the financial transaction of buying and selling items --/
structure Transaction where
  grinder_cp : ℝ  -- Cost price of grinder
  mobile_cp : ℝ   -- Cost price of mobile
  mobile_profit_percent : ℝ  -- Profit percentage on mobile
  total_profit : ℝ  -- Overall profit
  grinder_loss_percent : ℝ  -- Loss percentage on grinder (to be proved)

/-- Theorem stating the conditions and the result to be proved --/
theorem grinder_loss_percentage
  (t : Transaction)
  (h1 : t.grinder_cp = 15000)
  (h2 : t.mobile_cp = 8000)
  (h3 : t.mobile_profit_percent = 10)
  (h4 : t.total_profit = 500)
  : t.grinder_loss_percent = 2 := by
  sorry


end NUMINAMATH_CALUDE_grinder_loss_percentage_l225_22579


namespace NUMINAMATH_CALUDE_equation_solution_l225_22553

theorem equation_solution (a c x : ℝ) : 2 * x^2 + c^2 = (a + x)^2 → x = -a + c ∨ x = -a - c := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l225_22553


namespace NUMINAMATH_CALUDE_complex_abs_sum_l225_22595

theorem complex_abs_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 8*I) = Real.sqrt 34 + Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_sum_l225_22595


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l225_22546

theorem diophantine_equation_solution :
  ∀ (p a b c : ℕ),
    p.Prime →
    0 < a ∧ 0 < b ∧ 0 < c →
    73 * p^2 + 6 = 9 * a^2 + 17 * b^2 + 17 * c^2 →
    ((p = 2 ∧ a = 1 ∧ b = 4 ∧ c = 1) ∨ (p = 2 ∧ a = 1 ∧ b = 1 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l225_22546


namespace NUMINAMATH_CALUDE_norway_visitors_l225_22548

/-- Given a group of people with information about their visits to Iceland and Norway,
    calculate the number of people who visited Norway. -/
theorem norway_visitors
  (total : ℕ)
  (iceland : ℕ)
  (both : ℕ)
  (neither : ℕ)
  (h1 : total = 50)
  (h2 : iceland = 25)
  (h3 : both = 21)
  (h4 : neither = 23) :
  total = iceland + (norway : ℕ) - both + neither ∧ norway = 23 :=
by sorry

end NUMINAMATH_CALUDE_norway_visitors_l225_22548


namespace NUMINAMATH_CALUDE_f_properties_l225_22592

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x - (1 / π) * x^2 + cos x

theorem f_properties :
  (∀ x ∈ Set.Icc 0 (π / 2), Monotone f) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < π → f x1 = f x2 → 
    deriv f ((x1 + x2) / 2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l225_22592


namespace NUMINAMATH_CALUDE_equal_cost_layover_l225_22566

/-- Represents a city in the country -/
structure City where
  id : Nat

/-- Represents an airline operating in the country -/
structure Airline where
  id : Nat

/-- Represents the transportation network of the country -/
structure CountryNetwork where
  cities : Finset City
  airlines : Finset Airline
  connections : City → City → Finset Airline
  cost : City → City → ℚ

/-- The conditions of the problem -/
def ProblemConditions (network : CountryNetwork) : Prop :=
  (network.cities.card = 100) ∧
  (network.airlines.card = 146) ∧
  (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 → 
    network.connections c1 c2 ≠ ∅) ∧
  (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 → 
    network.cost c1 c2 = 1 / (network.connections c1 c2).card) ∧
  (∀ c1 c2 c3 : City, c1 ∈ network.cities → c2 ∈ network.cities → c3 ∈ network.cities → 
    c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 → 
    network.cost c1 c2 + network.cost c2 c3 ≥ network.cost c1 c3)

/-- The theorem to be proved -/
theorem equal_cost_layover (network : CountryNetwork) 
  (h : ProblemConditions network) : 
  ∃ c1 c2 c3 : City, c1 ∈ network.cities ∧ c2 ∈ network.cities ∧ c3 ∈ network.cities ∧
  c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
  network.cost c1 c2 = network.cost c2 c3 :=
sorry

end NUMINAMATH_CALUDE_equal_cost_layover_l225_22566


namespace NUMINAMATH_CALUDE_eggs_in_fridge_l225_22569

/-- Given a chef with eggs and cake-making information, calculate the number of eggs left in the fridge. -/
theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (cakes_made : ℕ) : 
  total_eggs = 60 → eggs_per_cake = 5 → cakes_made = 10 → 
  total_eggs - (eggs_per_cake * cakes_made) = 10 := by
  sorry

#check eggs_in_fridge

end NUMINAMATH_CALUDE_eggs_in_fridge_l225_22569


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l225_22565

/-- Given two parallel vectors a and b in R², where a = (1, 2) and b = (-2, y),
    prove that y must equal -4. -/
theorem parallel_vectors_y_value (y : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) →
  y = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l225_22565


namespace NUMINAMATH_CALUDE_integer_between_bounds_l225_22515

theorem integer_between_bounds (x : ℤ) :
  (-4.5 : ℝ) < (x : ℝ) ∧ (x : ℝ) < (-4 : ℝ) / 3 →
  x = -4 ∨ x = -3 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_integer_between_bounds_l225_22515


namespace NUMINAMATH_CALUDE_square_side_length_l225_22560

theorem square_side_length (area : ℚ) (side_length : ℚ) 
  (h1 : area = 9 / 16) 
  (h2 : side_length * side_length = area) : 
  side_length = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l225_22560


namespace NUMINAMATH_CALUDE_highest_number_on_paper_l225_22506

theorem highest_number_on_paper (n : ℕ) : 
  (1 : ℚ) / n = 0.01020408163265306 → n = 98 :=
by sorry

end NUMINAMATH_CALUDE_highest_number_on_paper_l225_22506


namespace NUMINAMATH_CALUDE_kite_area_in_square_l225_22586

/-- Given a 10 cm by 10 cm square with diagonals and a vertical line segment from
    the midpoint of the bottom side to the top side, the area of the kite-shaped
    region formed around the vertical line segment is 25 cm². -/
theorem kite_area_in_square (square_side : ℝ) (kite_area : ℝ) : 
  square_side = 10 → kite_area = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_kite_area_in_square_l225_22586


namespace NUMINAMATH_CALUDE_solution_k_value_l225_22524

theorem solution_k_value (x y k : ℝ) : 
  x = -3 ∧ y = 2 ∧ 2*x + k*y = 0 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_k_value_l225_22524


namespace NUMINAMATH_CALUDE_mortgage_loan_amount_l225_22590

/-- The mortgage loan problem -/
theorem mortgage_loan_amount 
  (initial_payment : ℝ) 
  (loan_percentage : ℝ) 
  (h1 : initial_payment = 2000000)
  (h2 : loan_percentage = 0.75) : 
  ∃ (total_cost : ℝ), 
    total_cost = initial_payment + loan_percentage * total_cost ∧ 
    loan_percentage * total_cost = 6000000 :=
by sorry

end NUMINAMATH_CALUDE_mortgage_loan_amount_l225_22590


namespace NUMINAMATH_CALUDE_train_length_problem_l225_22501

/-- Proves that under given conditions, the length of each train is 60 meters -/
theorem train_length_problem (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 48) (h2 : v_slow = 36) (h3 : t = 36) :
  let v_rel := v_fast - v_slow
  let d := v_rel * t * (5 / 18)
  d / 2 = 60 := by sorry

end NUMINAMATH_CALUDE_train_length_problem_l225_22501


namespace NUMINAMATH_CALUDE_division_of_decimals_l225_22572

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l225_22572


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l225_22530

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (6 + Real.sqrt (36 - 32)) / 2
  let r₂ := (6 - Real.sqrt (36 - 32)) / 2
  r₁ + r₂ = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l225_22530


namespace NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l225_22538

theorem xy_squared_minus_x_squared_y (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x * y = 3) : 
  x * y^2 - x^2 * y = -6 := by sorry

end NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l225_22538


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l225_22525

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l225_22525


namespace NUMINAMATH_CALUDE_closed_set_properties_l225_22541

-- Define a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-4, -2, 0, 2, 4}
def M : Set Int := {-4, -2, 0, 2, 4}

-- Define the set of positive integers
def positive_integers : Set Int := {n : Int | n > 0}

-- Define a general closed set
def closed_set (A : Set Int) : Prop := is_closed_set A

theorem closed_set_properties :
  (¬ is_closed_set M) ∧
  (¬ is_closed_set positive_integers) ∧
  (∃ A₁ A₂ : Set Int, closed_set A₁ ∧ closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂)) :=
sorry

end NUMINAMATH_CALUDE_closed_set_properties_l225_22541


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l225_22512

theorem abs_sum_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l225_22512


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l225_22573

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 5) : 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 5 → 
    9/x + 16/y + 25/z ≤ 9/a + 16/b + 25/c) ∧
  9/x + 16/y + 25/z = 28.8 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l225_22573


namespace NUMINAMATH_CALUDE_area_AEC_is_18_l225_22543

-- Define the lengths BE and EC
def BE : ℝ := 3
def EC : ℝ := 2

-- Define the area of triangle ABE
def area_ABE : ℝ := 27

-- Theorem statement
theorem area_AEC_is_18 :
  let ratio := BE / EC
  let area_AEC := (EC / BE) * area_ABE
  area_AEC = 18 := by sorry

end NUMINAMATH_CALUDE_area_AEC_is_18_l225_22543


namespace NUMINAMATH_CALUDE_cat_stairs_ways_l225_22575

def stair_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | m + 4 => stair_ways m + stair_ways (m + 1) + stair_ways (m + 2)

theorem cat_stairs_ways :
  stair_ways 10 = 12 :=
by sorry

end NUMINAMATH_CALUDE_cat_stairs_ways_l225_22575


namespace NUMINAMATH_CALUDE_cube_side_length_l225_22591

theorem cube_side_length (v : Real) (s : Real) :
  v = 8 →
  v = s^3 →
  ∃ (x : Real), 
    6 * x^2 = 3 * (6 * s^2) ∧
    x = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_l225_22591


namespace NUMINAMATH_CALUDE_dave_outer_space_books_l225_22504

/-- The number of books about outer space Dave bought -/
def outer_space_books : ℕ := 6

/-- The number of books about animals Dave bought -/
def animal_books : ℕ := 8

/-- The number of books about trains Dave bought -/
def train_books : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 6

/-- The total amount Dave spent on books in dollars -/
def total_spent : ℕ := 102

theorem dave_outer_space_books :
  outer_space_books = (total_spent - book_cost * (animal_books + train_books)) / book_cost :=
by sorry

end NUMINAMATH_CALUDE_dave_outer_space_books_l225_22504


namespace NUMINAMATH_CALUDE_expression_simplification_l225_22581

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b > 0) (hab : a^(1/3) * b^(1/4) ≠ 2) :
  ((a^2 * b * Real.sqrt b - 6 * a^(5/3) * b^(5/4) + 12 * a * b * a^(1/3) - 8 * a * b^(3/4))^(2/3)) /
  (a * b * a^(1/3) - 4 * a * b^(3/4) + 4 * a^(2/3) * Real.sqrt b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l225_22581


namespace NUMINAMATH_CALUDE_largest_repeated_product_365_l225_22531

def is_eight_digit_repeated (n : ℕ) : Prop :=
  100000000 > n ∧ n ≥ 10000000 ∧ 
  ∃ (a b c d : ℕ), n = a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + 
                    a * 1000 + b * 100 + c * 10 + d

theorem largest_repeated_product_365 : 
  (∀ m : ℕ, m > 273863 → ¬(is_eight_digit_repeated (m * 365))) ∧ 
  is_eight_digit_repeated (273863 * 365) := by
sorry

#eval 273863 * 365  -- Should output 99959995

end NUMINAMATH_CALUDE_largest_repeated_product_365_l225_22531


namespace NUMINAMATH_CALUDE_Q_one_smallest_l225_22556

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 - 3*x^2 + 6*x - 5

theorem Q_one_smallest : 
  let q1 := Q 1
  let prod_zeros := -5
  let sum_coeff := 1 + (-2) + (-3) + 6 + (-5)
  q1 ≤ prod_zeros ∧ q1 ≤ sum_coeff :=
by sorry

end NUMINAMATH_CALUDE_Q_one_smallest_l225_22556


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l225_22549

theorem unique_triplet_solution :
  ∃! (m n p : ℕ+), 
    Nat.Prime p ∧ 
    (2 : ℕ)^(m : ℕ) * (p : ℕ)^2 + 1 = (n : ℕ)^5 ∧
    m = 1 ∧ n = 3 ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l225_22549


namespace NUMINAMATH_CALUDE_kevins_age_exists_and_unique_l225_22599

theorem kevins_age_exists_and_unique :
  ∃! x : ℕ, 
    0 < x ∧ 
    x ≤ 120 ∧ 
    ∃ y : ℕ, x - 2 = y^2 ∧
    ∃ z : ℕ, x + 2 = z^3 := by
  sorry

end NUMINAMATH_CALUDE_kevins_age_exists_and_unique_l225_22599


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l225_22534

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l225_22534


namespace NUMINAMATH_CALUDE_nested_g_equals_cos_fifteen_fourths_l225_22500

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x / 2)

theorem nested_g_equals_cos_fifteen_fourths :
  0 < (1 : ℝ) / 2 ∧
  (∀ x : ℝ, 0 < x → 0 < g x) →
  g (g (g (g (g ((1 : ℝ) / 2) + 1) + 1) + 1) + 1) = Real.cos (15 / 4 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_nested_g_equals_cos_fifteen_fourths_l225_22500


namespace NUMINAMATH_CALUDE_wall_bricks_count_l225_22552

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 1800

/-- Time taken by the first bricklayer to build the wall alone -/
def time_bricklayer1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time_bricklayer2 : ℕ := 12

/-- Reduction in combined output when working together -/
def output_reduction : ℕ := 15

/-- Time taken to complete the wall when working together -/
def time_together : ℕ := 5

theorem wall_bricks_count :
  (time_together : ℝ) * ((total_bricks / time_bricklayer1 : ℝ) +
  (total_bricks / time_bricklayer2 : ℝ) - output_reduction) = total_bricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l225_22552


namespace NUMINAMATH_CALUDE_equation_system_solvability_l225_22562

theorem equation_system_solvability : ∃ (x y z : ℝ), 
  (2 * x + y = 4) ∧ 
  (x^2 + 3 * y = 5) ∧ 
  (3 * x - 1.5 * y + z = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solvability_l225_22562


namespace NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l225_22551

/-- The decimal representation of 5/17 has a repetend of 294117647058823529 -/
theorem repetend_of_five_seventeenths :
  ∃ (n : ℕ), (5 : ℚ) / 17 = (n : ℚ) / 999999999999999999 ∧
  n = 294117647058823529 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_five_seventeenths_l225_22551


namespace NUMINAMATH_CALUDE_work_rate_ratio_l225_22516

/-- 
Theorem: Given a job that P can complete in 4 days, and P and Q together can complete in 3 days, 
the ratio of Q's work rate to P's work rate is 1/3.
-/
theorem work_rate_ratio (p q : ℝ) : 
  p > 0 ∧ q > 0 →  -- Ensure positive work rates
  (1 / p = 1 / 4) →  -- P completes the job in 4 days
  (1 / (p + q) = 1 / 3) →  -- P and Q together complete the job in 3 days
  q / p = 1 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_work_rate_ratio_l225_22516


namespace NUMINAMATH_CALUDE_solve_for_m_l225_22521

theorem solve_for_m : ∀ m : ℚ, 
  (∃ x y : ℚ, m * x + y = 2 ∧ x = -2 ∧ y = 1) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l225_22521


namespace NUMINAMATH_CALUDE_ratio_fraction_value_l225_22542

theorem ratio_fraction_value (a b : ℝ) (h : a / b = 4) :
  (a - 3 * b) / (2 * a - b) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_fraction_value_l225_22542


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l225_22557

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l225_22557


namespace NUMINAMATH_CALUDE_pipe_b_fill_time_l225_22598

/-- Given a tank and three pipes A, B, and C, prove that pipe B fills the tank in 4 hours. -/
theorem pipe_b_fill_time (fill_time_A fill_time_B empty_time_C all_pipes_time : ℝ) 
  (h1 : fill_time_A = 3)
  (h2 : empty_time_C = 4)
  (h3 : all_pipes_time = 3.000000000000001)
  (h4 : 1 / fill_time_A + 1 / fill_time_B - 1 / empty_time_C = 1 / all_pipes_time) :
  fill_time_B = 4 := by
sorry

end NUMINAMATH_CALUDE_pipe_b_fill_time_l225_22598


namespace NUMINAMATH_CALUDE_opposite_values_imply_a_half_l225_22523

theorem opposite_values_imply_a_half (a : ℚ) : (2 * a) + (1 - 4 * a) = 0 → a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_values_imply_a_half_l225_22523


namespace NUMINAMATH_CALUDE_total_weight_AlI3_is_3261_44_l225_22561

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of aluminum atoms in a molecule of AlI3 -/
def num_Al_atoms : ℕ := 1

/-- The number of iodine atoms in a molecule of AlI3 -/
def num_I_atoms : ℕ := 3

/-- The number of moles of AlI3 -/
def num_moles : ℝ := 8

/-- The total weight of AlI3 in grams -/
def total_weight_AlI3 : ℝ :=
  num_moles * (num_Al_atoms * atomic_weight_Al + num_I_atoms * atomic_weight_I)

theorem total_weight_AlI3_is_3261_44 : total_weight_AlI3 = 3261.44 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_AlI3_is_3261_44_l225_22561


namespace NUMINAMATH_CALUDE_ladder_problem_l225_22507

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : ℝ, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l225_22507


namespace NUMINAMATH_CALUDE_mathematics_puzzle_solution_l225_22559

/-- Represents a mapping from characters to either digits or arithmetic operations -/
def LetterMapping := Char → Option (Nat ⊕ Bool)

/-- The word to be mapped -/
def word : List Char := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']

/-- Evaluates an expression given a mapping -/
def evalExpression (mapping : LetterMapping) (expr : List Char) : Option Int := sorry

/-- Checks if a mapping is valid according to the problem constraints -/
def isValidMapping (mapping : LetterMapping) : Prop := sorry

theorem mathematics_puzzle_solution :
  ∃ (mapping : LetterMapping),
    isValidMapping mapping ∧
    evalExpression mapping word = some 2014 := by sorry

end NUMINAMATH_CALUDE_mathematics_puzzle_solution_l225_22559


namespace NUMINAMATH_CALUDE_divisible_by_nineteen_l225_22519

theorem divisible_by_nineteen (n : ℕ) :
  ∃ k : ℤ, (5 : ℤ)^(2*n+1) * 2^(n+2) + 3^(n+2) * 2^(2*n+1) = 19 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nineteen_l225_22519


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l225_22535

theorem gcd_special_numbers : Nat.gcd 777777777 222222222222 = 999 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l225_22535


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l225_22537

theorem decimal_to_fraction :
  (0.32 : ℚ) = 8 / 25 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l225_22537


namespace NUMINAMATH_CALUDE_fraction_simplification_l225_22503

theorem fraction_simplification (x : ℝ) (h : x = 3) : 
  (x^8 + 16*x^4 + 64 + 4*x^2) / (x^4 + 8) = 89 + 36/89 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l225_22503


namespace NUMINAMATH_CALUDE_bounded_region_area_l225_22554

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 30*|x| = 360

-- Define the vertices of the parallelogram
def vertices : List (ℝ × ℝ) :=
  [(0, -30), (0, 30), (15, -30), (-15, 30)]

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ graph_equation x y}

-- Theorem statement
theorem bounded_region_area :
  MeasureTheory.volume (bounded_region) = 1800 :=
sorry

end NUMINAMATH_CALUDE_bounded_region_area_l225_22554


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l225_22570

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 + 4 * p^2 - 5 * p - 6 = 0) →
  (3 * q^3 + 4 * q^2 - 5 * q - 6 = 0) →
  (3 * r^3 + 4 * r^2 - 5 * r - 6 = 0) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l225_22570


namespace NUMINAMATH_CALUDE_check_tianning_pairs_find_x_for_negative_five_evaluate_expression_for_tianning_pair_l225_22564

-- Define Tianning pair
def is_tianning_pair (a b : ℝ) : Prop := a + b = a * b

-- Theorem 1: Checking specific pairs
theorem check_tianning_pairs :
  is_tianning_pair 3 1.5 ∧
  is_tianning_pair (-1/2) (1/3) ∧
  ¬ is_tianning_pair (3/4) 1 :=
sorry

-- Theorem 2: Finding x for (-5, x)
theorem find_x_for_negative_five :
  ∃ x, is_tianning_pair (-5) x ∧ x = 5/6 :=
sorry

-- Theorem 3: Evaluating expression for any Tianning pair
theorem evaluate_expression_for_tianning_pair (m n : ℝ) :
  is_tianning_pair m n →
  4*(m*n+m-2*(m*n-3))-2*(3*m^2-2*n)+6*m^2 = 24 :=
sorry

end NUMINAMATH_CALUDE_check_tianning_pairs_find_x_for_negative_five_evaluate_expression_for_tianning_pair_l225_22564


namespace NUMINAMATH_CALUDE_triangle_angles_theorem_l225_22511

/-- A triangle with vertices A, B, and C -/
structure Triangle (α : Type*) :=
  (A B C : α)

/-- The altitudes of a triangle -/
structure Altitudes (α : Type*) :=
  (AA₁ BB₁ CC₁ : α × α)

/-- Similarity of triangles -/
def similar {α : Type*} (t1 t2 : Triangle α) : Prop := sorry

/-- The angles of a triangle -/
def angles {α : Type*} (t : Triangle α) : ℝ × ℝ × ℝ := sorry

theorem triangle_angles_theorem {α : Type*} (ABC : Triangle α) (A₁B₁C₁ : Triangle α) (alt : Altitudes α) :
  (alt.AA₁.1 = ABC.A ∧ alt.BB₁.1 = ABC.B ∧ alt.CC₁.1 = ABC.C) →
  similar ABC A₁B₁C₁ →
  (angles ABC = (60, 60, 60) ∨ angles ABC = (720/7, 360/7, 180/7)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_theorem_l225_22511


namespace NUMINAMATH_CALUDE_valid_n_count_l225_22545

-- Define the triangle sides as functions of n
def AB (n : ℕ) : ℕ := 3 * n + 6
def BC (n : ℕ) : ℕ := 2 * n + 15
def AC (n : ℕ) : ℕ := 2 * n + 5

-- Define the conditions for a valid triangle
def isValidTriangle (n : ℕ) : Prop :=
  AB n + BC n > AC n ∧
  AB n + AC n > BC n ∧
  BC n + AC n > AB n ∧
  BC n > AB n ∧
  AB n > AC n

-- Theorem stating that there are exactly 7 valid values for n
theorem valid_n_count :
  ∃! (s : Finset ℕ), s.card = 7 ∧ ∀ n, n ∈ s ↔ isValidTriangle n :=
sorry

end NUMINAMATH_CALUDE_valid_n_count_l225_22545


namespace NUMINAMATH_CALUDE_total_red_stripes_l225_22596

/-- Calculates the number of red stripes in Flag A -/
def red_stripes_a (total_stripes : ℕ) : ℕ :=
  1 + (total_stripes - 1) / 2

/-- Calculates the number of red stripes in Flag B -/
def red_stripes_b (total_stripes : ℕ) : ℕ :=
  total_stripes / 3

/-- Calculates the number of red stripes in Flag C -/
def red_stripes_c (total_stripes : ℕ) : ℕ :=
  let full_patterns := total_stripes / 9
  let remaining_stripes := total_stripes % 9
  2 * full_patterns + min remaining_stripes 2

/-- The main theorem stating the total number of red stripes -/
theorem total_red_stripes :
  let flag_a_count := 20
  let flag_b_count := 30
  let flag_c_count := 40
  let flag_a_stripes := 30
  let flag_b_stripes := 45
  let flag_c_stripes := 60
  flag_a_count * red_stripes_a flag_a_stripes +
  flag_b_count * red_stripes_b flag_b_stripes +
  flag_c_count * red_stripes_c flag_c_stripes = 1310 := by
  sorry

end NUMINAMATH_CALUDE_total_red_stripes_l225_22596


namespace NUMINAMATH_CALUDE_long_tennis_players_l225_22558

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 35 →
  football = 26 →
  both = 17 →
  neither = 6 →
  ∃ long_tennis : ℕ, long_tennis = 20 ∧ 
    long_tennis = total - (football - both) - neither :=
by sorry

end NUMINAMATH_CALUDE_long_tennis_players_l225_22558


namespace NUMINAMATH_CALUDE_metal_bar_weight_l225_22539

/-- Represents the properties of a metal alloy bar --/
structure MetalBar where
  tin_weight : ℝ
  silver_weight : ℝ
  total_weight_loss : ℝ
  tin_loss_rate : ℝ
  silver_loss_rate : ℝ
  tin_silver_ratio : ℝ

/-- Theorem stating the weight of the metal bar given the conditions --/
theorem metal_bar_weight (bar : MetalBar)
  (h1 : bar.total_weight_loss = 6)
  (h2 : bar.tin_loss_rate = 1.375 / 10)
  (h3 : bar.silver_loss_rate = 0.375 / 5)
  (h4 : bar.tin_silver_ratio = 2 / 3)
  (h5 : bar.tin_weight * bar.tin_loss_rate + bar.silver_weight * bar.silver_loss_rate = bar.total_weight_loss)
  (h6 : bar.tin_weight / bar.silver_weight = bar.tin_silver_ratio) :
  bar.tin_weight + bar.silver_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_metal_bar_weight_l225_22539


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l225_22580

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ (2 / x = 1 / (x + 1)) ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l225_22580


namespace NUMINAMATH_CALUDE_car_journey_theorem_l225_22578

theorem car_journey_theorem (local_distance : ℝ) (local_speed : ℝ) (highway_speed : ℝ) (average_speed : ℝ) (highway_distance : ℝ) :
  local_distance = 60 ∧
  local_speed = 20 ∧
  highway_speed = 60 ∧
  average_speed = 36 ∧
  average_speed = (local_distance + highway_distance) / (local_distance / local_speed + highway_distance / highway_speed) →
  highway_distance = 120 := by
sorry

end NUMINAMATH_CALUDE_car_journey_theorem_l225_22578


namespace NUMINAMATH_CALUDE_mirror_area_l225_22517

/-- The area of a rectangular mirror inside a frame with given external dimensions and frame width -/
theorem mirror_area (frame_height frame_width frame_side_width : ℝ) :
  frame_height = 100 ∧ 
  frame_width = 140 ∧ 
  frame_side_width = 15 →
  (frame_height - 2 * frame_side_width) * (frame_width - 2 * frame_side_width) = 7700 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l225_22517


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l225_22587

theorem arithmetic_calculation : 6 / (-3) + 2^2 * (1 - 4) = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l225_22587


namespace NUMINAMATH_CALUDE_common_ratio_is_negative_half_l225_22502

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  not_constant : ∃ i j, a i ≠ a j
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  a_3 : a 3 = 5 / 2
  S_3 : (a 1) + (a 2) + (a 3) = 15 / 2

/-- The common ratio of the geometric sequence -/
def common_ratio (seq : GeometricSequence) : ℚ := seq.a 2 / seq.a 1

theorem common_ratio_is_negative_half (seq : GeometricSequence) : 
  common_ratio seq = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_negative_half_l225_22502


namespace NUMINAMATH_CALUDE_solution_pairs_l225_22571

theorem solution_pairs : ∀ x y : ℝ,
  (x^2 + y^2 - 48*x - 29*y + 714 = 0 ∧
   2*x*y - 29*x - 48*y + 756 = 0) ↔
  ((x = 31.5 ∧ y = 10.5) ∨
   (x = 20 ∧ y = 22) ∨
   (x = 28 ∧ y = 7) ∨
   (x = 16.5 ∧ y = 18.5)) := by
sorry

end NUMINAMATH_CALUDE_solution_pairs_l225_22571


namespace NUMINAMATH_CALUDE_general_admission_tickets_l225_22508

theorem general_admission_tickets (student_price general_price total_tickets total_money : ℕ) 
  (h1 : student_price = 4)
  (h2 : general_price = 6)
  (h3 : total_tickets = 525)
  (h4 : total_money = 2876) :
  ∃ (student_tickets general_tickets : ℕ),
    student_tickets + general_tickets = total_tickets ∧
    student_tickets * student_price + general_tickets * general_price = total_money ∧
    general_tickets = 388 := by
  sorry

end NUMINAMATH_CALUDE_general_admission_tickets_l225_22508
