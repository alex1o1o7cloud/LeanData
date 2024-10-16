import Mathlib

namespace NUMINAMATH_CALUDE_andy_final_position_l168_16807

/-- Represents a position on a 2D plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Andy's movement function -/
def move (pos : Position) (dir : Direction) (steps : Nat) : Position :=
  match dir with
  | Direction.East  => { x := pos.x + steps, y := pos.y }
  | Direction.North => { x := pos.x, y := pos.y + steps }
  | Direction.West  => { x := pos.x - steps, y := pos.y }
  | Direction.South => { x := pos.x, y := pos.y - steps }

/-- Function to apply wind effect -/
def applyWind (pos : Position) : Position :=
  { x := pos.x, y := pos.y + 1 }

/-- Andy's full movement pattern -/
def andyMove (start : Position) (turns : Nat) : Position :=
  sorry

/-- Theorem stating Andy's final position after 1022 turns -/
theorem andy_final_position :
  andyMove { x := -30, y := 30 } 1022 = { x := -1562, y := 881 } :=
  sorry

end NUMINAMATH_CALUDE_andy_final_position_l168_16807


namespace NUMINAMATH_CALUDE_correct_zongzi_profit_equation_l168_16808

/-- Represents the daily profit equation for a supermarket selling zongzi -/
def zongzi_profit_equation (initial_price cost_price initial_sales sales_increase target_profit : ℝ) : Prop :=
  ∀ x : ℝ, (initial_price - x - cost_price) * (initial_sales + sales_increase * x) = target_profit

/-- Theorem stating the correct profit equation for the given zongzi selling scenario -/
theorem correct_zongzi_profit_equation :
  zongzi_profit_equation 16 10 200 80 1440 :=
sorry

end NUMINAMATH_CALUDE_correct_zongzi_profit_equation_l168_16808


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l168_16833

/-- A quadratic function with vertex (-3, 2) passing through (2, -43) has a = -9/5 --/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x + 3)^2 + 2) →  -- vertex form
  (a * 2^2 + b * 2 + c = -43) →                       -- passes through (2, -43)
  a = -9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l168_16833


namespace NUMINAMATH_CALUDE_cost_per_cow_l168_16813

/-- Calculates the cost per cow given Timothy's expenses --/
theorem cost_per_cow (land_acres : ℕ) (land_cost_per_acre : ℕ)
  (house_cost : ℕ) (num_cows : ℕ) (num_chickens : ℕ)
  (chicken_cost : ℕ) (solar_install_hours : ℕ)
  (solar_install_rate : ℕ) (solar_equipment_cost : ℕ)
  (total_cost : ℕ) :
  land_acres = 30 →
  land_cost_per_acre = 20 →
  house_cost = 120000 →
  num_cows = 20 →
  num_chickens = 100 →
  chicken_cost = 5 →
  solar_install_hours = 6 →
  solar_install_rate = 100 →
  solar_equipment_cost = 6000 →
  total_cost = 147700 →
  (total_cost - (land_acres * land_cost_per_acre + house_cost + 
    num_chickens * chicken_cost + 
    solar_install_hours * solar_install_rate + solar_equipment_cost)) / num_cows = 1000 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_cow_l168_16813


namespace NUMINAMATH_CALUDE_allocation_methods_l168_16852

/-- The number of warriors in the class -/
def total_warriors : ℕ := 6

/-- The number of tasks to be completed -/
def num_tasks : ℕ := 4

/-- The number of leadership positions (captain and vice-captain) -/
def leadership_positions : ℕ := 2

/-- The number of participating warriors -/
def participating_warriors : ℕ := 4

theorem allocation_methods :
  (leadership_positions.choose 1) *
  ((total_warriors - leadership_positions).choose (participating_warriors - 1)) *
  (participating_warriors.factorial) = 192 :=
sorry

end NUMINAMATH_CALUDE_allocation_methods_l168_16852


namespace NUMINAMATH_CALUDE_square_sum_equals_37_l168_16877

theorem square_sum_equals_37 (x y : ℝ) 
  (h1 : y + 3 = (x - 3)^2)
  (h2 : x + 3 = (y - 3)^2)
  (h3 : x ≠ y) :
  x^2 + y^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_37_l168_16877


namespace NUMINAMATH_CALUDE_jasmine_laps_l168_16899

/-- Calculates the total number of laps swum in a given number of weeks -/
def total_laps (laps_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * num_weeks

/-- Proves that Jasmine swims 300 laps in five weeks -/
theorem jasmine_laps : total_laps 12 5 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_laps_l168_16899


namespace NUMINAMATH_CALUDE_inscribed_ngon_existence_l168_16843

/-- An n-gon inscribed in a circle with sides parallel to n given lines -/
structure InscribedNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ
  lines : Fin n → ℝ × ℝ → Prop

/-- The measure of the angle at a vertex of the n-gon -/
def angle (ngon : InscribedNGon n) (i : Fin n) : ℝ := sorry

/-- The sum of odd-indexed angles -/
def sumOddAngles (ngon : InscribedNGon n) : ℝ := sorry

/-- The sum of even-indexed angles -/
def sumEvenAngles (ngon : InscribedNGon n) : ℝ := sorry

/-- The existence of an inscribed n-gon with sides parallel to given lines -/
def existsInscribedNGon (n : ℕ) (center : ℝ × ℝ) (radius : ℝ) (lines : Fin n → ℝ × ℝ → Prop) : Prop := sorry

theorem inscribed_ngon_existence (n : ℕ) (center : ℝ × ℝ) (radius : ℝ) (lines : Fin n → ℝ × ℝ → Prop) :
  (n % 2 = 1 ∧ existsInscribedNGon n center radius lines) ∨
  (n % 2 = 0 ∧ (existsInscribedNGon n center radius lines ↔
    ∃ (ngon : InscribedNGon n), sumOddAngles ngon = sumEvenAngles ngon)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_ngon_existence_l168_16843


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l168_16819

/-- The sampling interval for systematic sampling -/
def sampling_interval (population_size : ℕ) (sample_size : ℕ) : ℕ :=
  population_size / sample_size

/-- Theorem: The sampling interval for a population of 1000 and sample size of 20 is 50 -/
theorem systematic_sampling_interval :
  sampling_interval 1000 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l168_16819


namespace NUMINAMATH_CALUDE_solve_equation_l168_16811

theorem solve_equation : ∃ x : ℝ, (x - 5)^4 = (1/16)⁻¹ ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l168_16811


namespace NUMINAMATH_CALUDE_strawberry_candies_count_candy_problem_l168_16876

theorem strawberry_candies_count : ℕ → ℕ → Prop :=
  fun total grape_diff =>
    ∀ (strawberry grape : ℕ),
      strawberry + grape = total →
      grape = strawberry - grape_diff →
      strawberry = 121

theorem candy_problem : strawberry_candies_count 240 2 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_candies_count_candy_problem_l168_16876


namespace NUMINAMATH_CALUDE_b_25_mod_35_l168_16874

/-- b_n is the integer obtained by writing all integers from 1 to n from left to right, each repeated twice -/
def b (n : ℕ) : ℕ :=
  -- Definition of b_n goes here
  sorry

/-- The remainder when b_25 is divided by 35 is 6 -/
theorem b_25_mod_35 : b 25 % 35 = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_25_mod_35_l168_16874


namespace NUMINAMATH_CALUDE_negative_two_plus_three_equals_one_l168_16816

theorem negative_two_plus_three_equals_one : (-2 : ℤ) + 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_plus_three_equals_one_l168_16816


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l168_16856

theorem theater_ticket_sales (adult_price child_price : ℕ) 
  (total_tickets adult_tickets child_tickets : ℕ) : 
  adult_price = 12 → 
  child_price = 4 → 
  total_tickets = 130 → 
  adult_tickets = 90 → 
  child_tickets = 40 → 
  adult_price * adult_tickets + child_price * child_tickets = 1240 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l168_16856


namespace NUMINAMATH_CALUDE_range_of_a_satisfying_equation_l168_16871

open Real

theorem range_of_a_satisfying_equation :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 3 * x + a * (2 * y - 4 * ℯ * x) * (log y - log x) = 0) ↔ 
  (a < 0 ∨ a ≥ 3 / (2 * ℯ)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_satisfying_equation_l168_16871


namespace NUMINAMATH_CALUDE_zeros_imply_a_range_l168_16892

/-- The function h(x) = ax² - x - ln(x) has two distinct zeros -/
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
  a * x₁^2 - x₁ - Real.log x₁ = 0 ∧
  a * x₂^2 - x₂ - Real.log x₂ = 0

/-- If h(x) has two distinct zeros, then 0 < a < 1 -/
theorem zeros_imply_a_range (a : ℝ) (h : a ≠ 0) :
  has_two_distinct_zeros a → 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_zeros_imply_a_range_l168_16892


namespace NUMINAMATH_CALUDE_jake_initial_cats_jake_initial_cats_is_one_l168_16846

/-- Proves that Jake initially had 1 cat given the conditions of the problem -/
theorem jake_initial_cats : ℝ → Prop :=
  fun initial_cats =>
    let food_per_cat : ℝ := 0.5
    let total_food_after : ℝ := 0.9
    let extra_food : ℝ := 0.4
    (initial_cats * food_per_cat + food_per_cat = total_food_after) ∧
    (food_per_cat = extra_food) →
    initial_cats = 1

/-- The theorem is true -/
theorem jake_initial_cats_is_one : jake_initial_cats 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_initial_cats_jake_initial_cats_is_one_l168_16846


namespace NUMINAMATH_CALUDE_problem_solution_l168_16860

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4) :
  x = 24^(2/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l168_16860


namespace NUMINAMATH_CALUDE_sum_of_numbers_l168_16835

theorem sum_of_numbers (x y : ℝ) : y = 2 * x - 3 ∧ y = 33 → x + y = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l168_16835


namespace NUMINAMATH_CALUDE_min_sum_squares_addends_of_18_l168_16859

theorem min_sum_squares_addends_of_18 :
  ∀ x y : ℝ, x + y = 18 → x^2 + y^2 ≥ 2 * 9^2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_addends_of_18_l168_16859


namespace NUMINAMATH_CALUDE_zions_dad_age_difference_l168_16810

/-- Proves that Zion's dad's age is 3 years more than 4 times Zion's age given the conditions. -/
theorem zions_dad_age_difference (zion_age : ℕ) (dad_age : ℕ) : 
  zion_age = 8 →
  dad_age > 4 * zion_age →
  dad_age + 10 = (zion_age + 10) + 27 →
  dad_age = 4 * zion_age + 3 := by
sorry

end NUMINAMATH_CALUDE_zions_dad_age_difference_l168_16810


namespace NUMINAMATH_CALUDE_brendas_mice_problem_l168_16882

theorem brendas_mice_problem (total_litters : Nat) (mice_per_litter : Nat) 
  (fraction_to_robbie : Rat) (multiplier_to_pet_store : Nat) (fraction_to_feeder : Rat) :
  total_litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1 / 6 →
  multiplier_to_pet_store = 3 →
  fraction_to_feeder = 1 / 2 →
  (total_litters * mice_per_litter 
    - (total_litters * mice_per_litter : Rat) * fraction_to_robbie 
    - (total_litters * mice_per_litter : Rat) * fraction_to_robbie * multiplier_to_pet_store) 
    * (1 - fraction_to_feeder) = 4 := by
  sorry

end NUMINAMATH_CALUDE_brendas_mice_problem_l168_16882


namespace NUMINAMATH_CALUDE_total_distance_biking_and_jogging_l168_16895

theorem total_distance_biking_and_jogging 
  (total_time : ℝ) 
  (biking_time : ℝ) 
  (biking_rate : ℝ) 
  (jogging_time : ℝ) 
  (jogging_rate : ℝ) 
  (h1 : total_time = 1.75) -- 1 hour and 45 minutes
  (h2 : biking_time = 1) -- 60 minutes in hours
  (h3 : biking_rate = 12)
  (h4 : jogging_time = 0.75) -- 45 minutes in hours
  (h5 : jogging_rate = 6) : 
  biking_rate * biking_time + jogging_rate * jogging_time = 16.5 := by
  sorry

#check total_distance_biking_and_jogging

end NUMINAMATH_CALUDE_total_distance_biking_and_jogging_l168_16895


namespace NUMINAMATH_CALUDE_total_tires_changed_is_304_l168_16805

/-- Represents the number of tires changed by Mike in a day -/
def total_tires_changed : ℕ :=
  let motorcycles := 12
  let cars := 10
  let bicycles := 8
  let trucks := 5
  let atvs := 7
  let dual_axle_trailers := 4
  let triple_axle_boat_trailers := 3
  let unicycles := 2
  let dually_pickup_trucks := 6

  let motorcycle_tires := 2
  let car_tires := 4
  let bicycle_tires := 2
  let truck_tires := 18
  let atv_tires := 4
  let dual_axle_trailer_tires := 8
  let triple_axle_boat_trailer_tires := 12
  let unicycle_tires := 1
  let dually_pickup_truck_tires := 6

  motorcycles * motorcycle_tires +
  cars * car_tires +
  bicycles * bicycle_tires +
  trucks * truck_tires +
  atvs * atv_tires +
  dual_axle_trailers * dual_axle_trailer_tires +
  triple_axle_boat_trailers * triple_axle_boat_trailer_tires +
  unicycles * unicycle_tires +
  dually_pickup_trucks * dually_pickup_truck_tires

/-- Theorem stating that the total number of tires changed by Mike in a day is 304 -/
theorem total_tires_changed_is_304 : total_tires_changed = 304 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_changed_is_304_l168_16805


namespace NUMINAMATH_CALUDE_tiffany_cans_l168_16803

theorem tiffany_cans (initial_bags : ℕ) (next_day_bags : ℕ) (total_bags : ℕ) :
  initial_bags = 10 →
  next_day_bags = 3 →
  total_bags = 20 →
  total_bags - (initial_bags + next_day_bags) = 7 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_cans_l168_16803


namespace NUMINAMATH_CALUDE_arithmetic_sequence_perfect_squares_l168_16897

/-- An arithmetic sequence of natural numbers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n ↦ a + n * d

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The theorem stating that if an arithmetic sequence of natural numbers contains
    one perfect square, it contains infinitely many perfect squares -/
theorem arithmetic_sequence_perfect_squares
  (a d : ℕ) -- First term and common difference of the arithmetic sequence
  (h : ∃ n : ℕ, IsPerfectSquare (ArithmeticSequence a d n)) :
  ∀ m : ℕ, ∃ k > m, IsPerfectSquare (ArithmeticSequence a d k) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_perfect_squares_l168_16897


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l168_16839

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) → 
  n + (n + 1) = 43 :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l168_16839


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l168_16824

theorem no_alpha_sequence_exists : ¬ ∃ (α : ℝ) (a : ℕ → ℝ),
  (0 < α ∧ α < 1) ∧
  (∀ n, 0 < a n) ∧
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l168_16824


namespace NUMINAMATH_CALUDE_union_M_N_l168_16845

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1}

theorem union_M_N : M ∪ N = {x | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l168_16845


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l168_16864

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l168_16864


namespace NUMINAMATH_CALUDE_circle_equation_proof_l168_16820

-- Define a circle in R^2
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define when a circle is tangent to the x-axis
def TangentToXAxis (c : Set (ℝ × ℝ)) : Prop :=
  ∃ x : ℝ, (x, 0) ∈ c ∧ ∀ y : ℝ, y ≠ 0 → (x, y) ∉ c

theorem circle_equation_proof :
  let c : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = 4}
  c = Circle (0, 2) 2 ∧ TangentToXAxis c := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l168_16820


namespace NUMINAMATH_CALUDE_cricket_average_l168_16832

theorem cricket_average (initial_average : ℝ) : 
  (14 * initial_average + 140) / 15 = initial_average + 8 →
  initial_average + 8 = 28 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l168_16832


namespace NUMINAMATH_CALUDE_cube_root_equality_l168_16841

theorem cube_root_equality (m : ℝ) : 
  (9 + 9 / m) ^ (1/3) = 9 * (9 / m) ^ (1/3) → m = 728 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equality_l168_16841


namespace NUMINAMATH_CALUDE_line_points_count_l168_16868

theorem line_points_count (n : ℕ) 
  (point1 : ∃ (a b : ℕ), a * b = 80 ∧ a + b + 1 = n)
  (point2 : ∃ (c d : ℕ), c * d = 90 ∧ c + d + 1 = n) :
  n = 22 := by
sorry

end NUMINAMATH_CALUDE_line_points_count_l168_16868


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l168_16849

theorem floor_abs_negative_real : ⌊|(-56.7 : ℝ)|⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l168_16849


namespace NUMINAMATH_CALUDE_min_printers_purchase_l168_16842

theorem min_printers_purchase (cost1 cost2 : ℕ) (h1 : cost1 = 350) (h2 : cost2 = 200) :
  ∃ (x y : ℕ), 
    x * cost1 = y * cost2 ∧ 
    x + y = 11 ∧
    ∀ (a b : ℕ), a * cost1 = b * cost2 → a + b ≥ 11 :=
by sorry

end NUMINAMATH_CALUDE_min_printers_purchase_l168_16842


namespace NUMINAMATH_CALUDE_inequality_solution_set_l168_16869

theorem inequality_solution_set (x : ℝ) : 
  |x - 4| - |x + 1| < 3 ↔ x ∈ Set.Ioo (-1/2 : ℝ) 4 ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l168_16869


namespace NUMINAMATH_CALUDE_stephanie_oranges_l168_16848

/-- Represents the number of store visits -/
def store_visits : ℕ := 8

/-- Represents the total number of oranges bought -/
def total_oranges : ℕ := 16

/-- Represents the number of oranges bought per visit -/
def oranges_per_visit : ℕ := total_oranges / store_visits

/-- Theorem stating that Stephanie buys 2 oranges each time she goes to the store -/
theorem stephanie_oranges : oranges_per_visit = 2 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l168_16848


namespace NUMINAMATH_CALUDE_min_A_over_B_l168_16817

theorem min_A_over_B (A B x : ℝ) (hA : A > 0) (hB : B > 0) (hx : x > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = B) :
  A / B ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_A_over_B_l168_16817


namespace NUMINAMATH_CALUDE_function_difference_bound_l168_16830

theorem function_difference_bound 
  (f : Set.Icc 0 1 → ℝ)
  (h1 : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h2 : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_difference_bound_l168_16830


namespace NUMINAMATH_CALUDE_solution_x_squared_equals_three_l168_16847

theorem solution_x_squared_equals_three :
  ∀ x : ℝ, x^2 = 3 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_x_squared_equals_three_l168_16847


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l168_16801

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  -- Area of the trapezoid
  area : ℝ
  -- Condition that one base is twice the other
  base_ratio : Bool
  -- Point of intersection of diagonals
  O : Point
  -- Midpoint of base AD
  P : Point
  -- Points where BP and CP intersect the diagonals
  M : Point
  N : Point

/-- The area of triangle MON in a trapezoid with specific properties -/
def area_MON (t : Trapezoid) : Set ℝ :=
  {45/4, 36/5}

/-- Theorem stating the area of triangle MON in a trapezoid with given properties -/
theorem trapezoid_triangle_area (t : Trapezoid) 
  (h1 : t.area = 405) : 
  (area_MON t).Nonempty ∧ (∀ x ∈ area_MON t, x = 45/4 ∨ x = 36/5) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l168_16801


namespace NUMINAMATH_CALUDE_first_year_interest_rate_l168_16853

/-- Proves that the first-year interest rate is 4% given the problem conditions --/
theorem first_year_interest_rate 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (second_year_rate : ℝ) 
  (h1 : initial_amount = 4000)
  (h2 : final_amount = 4368)
  (h3 : second_year_rate = 0.05)
  : ∃ (R : ℝ), 
    initial_amount * (1 + R) * (1 + second_year_rate) = final_amount ∧ 
    R = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_first_year_interest_rate_l168_16853


namespace NUMINAMATH_CALUDE_not_monomial_two_over_a_l168_16888

/-- Definition of a monomial -/
def is_monomial (e : ℤ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ x, e x = c * x^n

/-- The expression 2/a is not a monomial -/
theorem not_monomial_two_over_a : ¬ is_monomial (λ a => 2 / a) := by
  sorry

end NUMINAMATH_CALUDE_not_monomial_two_over_a_l168_16888


namespace NUMINAMATH_CALUDE_topsoil_cost_for_8_cubic_yards_l168_16836

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 8

/-- The cost of topsoil for a given volume in cubic yards -/
def topsoil_cost (volume : ℝ) : ℝ :=
  volume * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_for_8_cubic_yards :
  topsoil_cost volume_in_cubic_yards = 1728 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_for_8_cubic_yards_l168_16836


namespace NUMINAMATH_CALUDE_clock_angle_at_9_l168_16851

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The number of degrees each hour represents on a clock face -/
def degrees_per_hour : ℕ := full_circle / clock_hours

/-- The position of the minute hand at 9:00 in degrees -/
def minute_hand_position : ℕ := 0

/-- The position of the hour hand at 9:00 in degrees -/
def hour_hand_position : ℕ := 9 * degrees_per_hour

/-- The smaller angle between the hour hand and minute hand at 9:00 -/
def smaller_angle : ℕ := min (hour_hand_position - minute_hand_position) (full_circle - (hour_hand_position - minute_hand_position))

theorem clock_angle_at_9 : smaller_angle = 90 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_9_l168_16851


namespace NUMINAMATH_CALUDE_kyler_wins_one_l168_16822

structure ChessTournament where
  peter_wins : ℕ
  peter_losses : ℕ
  emma_wins : ℕ
  emma_losses : ℕ
  kyler_losses : ℕ

def kyler_wins (t : ChessTournament) : ℕ :=
  (t.peter_wins + t.emma_wins + t.kyler_losses) - (t.peter_losses + t.emma_losses)

theorem kyler_wins_one (t : ChessTournament) 
  (h1 : t.peter_wins = 4) 
  (h2 : t.peter_losses = 2) 
  (h3 : t.emma_wins = 3) 
  (h4 : t.emma_losses = 3) 
  (h5 : t.kyler_losses = 3) : 
  kyler_wins t = 1 := by
  sorry

end NUMINAMATH_CALUDE_kyler_wins_one_l168_16822


namespace NUMINAMATH_CALUDE_day_care_ratio_l168_16891

/-- Proves that the initial ratio of toddlers to infants is 7:3 given the conditions of the day care problem. -/
theorem day_care_ratio (toddlers initial_infants : ℕ) : 
  toddlers = 42 →
  (toddlers : ℚ) / (initial_infants + 12 : ℚ) = 7 / 5 →
  (toddlers : ℚ) / (initial_infants : ℚ) = 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_day_care_ratio_l168_16891


namespace NUMINAMATH_CALUDE_simple_random_sampling_problem_l168_16855

/-- Prove that in a simple random sampling where 13 individuals are drawn one by one
    from a group of n individuals (n > 13), if the probability for each of the remaining
    individuals to be drawn on the second draw is 1/3, then n = 37. -/
theorem simple_random_sampling_problem (n : ℕ) (h1 : n > 13) :
  (12 : ℝ) / (n - 1 : ℝ) = (1 : ℝ) / 3 → n = 37 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_problem_l168_16855


namespace NUMINAMATH_CALUDE_roger_coins_count_l168_16854

/-- Calculates the total number of coins given the number of piles of quarters,
    piles of dimes, and coins per pile. -/
def totalCoins (quarterPiles dimePiles coinsPerPile : ℕ) : ℕ :=
  (quarterPiles + dimePiles) * coinsPerPile

/-- Theorem stating that with 3 piles of quarters, 3 piles of dimes,
    and 7 coins per pile, the total number of coins is 42. -/
theorem roger_coins_count :
  totalCoins 3 3 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_roger_coins_count_l168_16854


namespace NUMINAMATH_CALUDE_initial_maple_trees_count_l168_16870

/-- The number of maple trees to be planted -/
def trees_to_plant : ℕ := 9

/-- The final number of maple trees after planting -/
def final_maple_trees : ℕ := 11

/-- The initial number of maple trees in the park -/
def initial_maple_trees : ℕ := final_maple_trees - trees_to_plant

theorem initial_maple_trees_count : initial_maple_trees = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_maple_trees_count_l168_16870


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l168_16890

def fruit_problem (apple_price orange_price : ℚ) 
                  (total_fruits : ℕ) 
                  (initial_avg_price desired_avg_price : ℚ) : Prop :=
  let oranges_to_remove := 10
  let remaining_fruits := total_fruits - oranges_to_remove
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / remaining_fruits = desired_avg_price

theorem fruit_stand_problem :
  fruit_problem (40/100) (60/100) 20 (56/100) (52/100) :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l168_16890


namespace NUMINAMATH_CALUDE_find_number_l168_16840

theorem find_number : ∃ x : ℝ, 0.62 * x - 0.20 * 250 = 43 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l168_16840


namespace NUMINAMATH_CALUDE_rectangular_room_shorter_side_l168_16850

theorem rectangular_room_shorter_side
  (perimeter : ℝ)
  (area : ℝ)
  (h_perimeter : perimeter = 42)
  (h_area : area = 108)
  (h_rect : ∃ (length width : ℝ), length > 0 ∧ width > 0 ∧
            2 * (length + width) = perimeter ∧
            length * width = area) :
  ∃ (shorter_side : ℝ), shorter_side = 9 ∧
    ∃ (longer_side : ℝ), longer_side > shorter_side ∧
      2 * (shorter_side + longer_side) = perimeter ∧
      shorter_side * longer_side = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_room_shorter_side_l168_16850


namespace NUMINAMATH_CALUDE_solution_value_l168_16826

theorem solution_value (a : ℝ) (h : a^2 - 5*a - 1 = 0) : 3*a^2 - 15*a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l168_16826


namespace NUMINAMATH_CALUDE_football_outcomes_count_l168_16887

def FootballOutcome := Nat × Nat × Nat

def total_matches (outcome : FootballOutcome) : Nat :=
  outcome.1 + outcome.2.1 + outcome.2.2

def total_points (outcome : FootballOutcome) : Nat :=
  3 * outcome.1 + outcome.2.1

def is_valid_outcome (outcome : FootballOutcome) : Prop :=
  total_matches outcome = 14 ∧ total_points outcome = 19

theorem football_outcomes_count :
  ∃! n : Nat, ∃ outcomes : Finset FootballOutcome,
    outcomes.card = n ∧
    (∀ o : FootballOutcome, o ∈ outcomes ↔ is_valid_outcome o) ∧
    n = 4 := by sorry

end NUMINAMATH_CALUDE_football_outcomes_count_l168_16887


namespace NUMINAMATH_CALUDE_simplify_complex_root_expression_l168_16867

theorem simplify_complex_root_expression (a : ℝ) :
  (((a^16)^(1/8))^(1/4) + ((a^16)^(1/4))^(1/8))^2 = 4*a := by sorry

end NUMINAMATH_CALUDE_simplify_complex_root_expression_l168_16867


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l168_16804

-- Define the matrix expression evaluation rule
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the equation to solve
def equation (x : ℝ) : Prop :=
  matrix_value (3 * x) (x + 2) (x + 1) (2 * x) = 6

-- State the theorem
theorem matrix_equation_solution :
  ∃ x₁ x₂ : ℝ, x₁ = -2 + Real.sqrt 10 ∧ x₂ = -2 - Real.sqrt 10 ∧
  ∀ x : ℝ, equation x ↔ (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l168_16804


namespace NUMINAMATH_CALUDE_ceiling_negative_fraction_cubed_l168_16802

theorem ceiling_negative_fraction_cubed : ⌈(-7/4)^3⌉ = -5 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_fraction_cubed_l168_16802


namespace NUMINAMATH_CALUDE_saltwater_volume_l168_16863

/-- Proves that the initial volume of a saltwater solution is 160 gallons given specific conditions --/
theorem saltwater_volume : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.20 * x = x * 0.20) ∧ 
  (0.20 * x + 16 = (1/3) * (3/4 * x + 8 + 16)) ∧ 
  (x = 160) := by
sorry

end NUMINAMATH_CALUDE_saltwater_volume_l168_16863


namespace NUMINAMATH_CALUDE_ellipse_properties_l168_16875

/-- An ellipse with equation x^2/2 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- The circle with diameter F₁F₂ -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2) = 1}

/-- The line x + y - √2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 = Real.sqrt 2}

/-- The dot product of vectors PF₁ and PF₂ -/
def dotProduct (P : ℝ × ℝ) : ℝ :=
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2)

theorem ellipse_properties :
  (∀ p ∈ Circle, p ∈ Line) ∧
  (∀ P ∈ Ellipse, dotProduct P ≥ 0 ∧ ∃ Q ∈ Ellipse, dotProduct Q = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l168_16875


namespace NUMINAMATH_CALUDE_sqrt_expression_one_sqrt_expression_two_sqrt_expression_three_l168_16879

-- Problem 1
theorem sqrt_expression_one : 
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem sqrt_expression_two : 
  1 / Real.sqrt 24 + |Real.sqrt 6 - 3| + (1/2)⁻¹ - 2016^0 = 4 - 11 * Real.sqrt 6 / 12 := by sorry

-- Problem 3
theorem sqrt_expression_three : 
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2)^2 = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_one_sqrt_expression_two_sqrt_expression_three_l168_16879


namespace NUMINAMATH_CALUDE_angle_I_measures_138_l168_16800

/-- A convex pentagon with specific angle properties -/
structure ConvexPentagon where
  -- Angles in degrees
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ
  -- Angle sum in a pentagon is 540°
  sum_eq_540 : F + G + H + I + J = 540
  -- Angles F, G, and H are congruent
  F_eq_G : F = G
  G_eq_H : G = H
  -- Angles I and J are congruent
  I_eq_J : I = J
  -- Angle F is 50° less than angle I
  F_eq_I_minus_50 : F = I - 50

/-- Theorem: In a convex pentagon with the given properties, angle I measures 138° -/
theorem angle_I_measures_138 (p : ConvexPentagon) : p.I = 138 := by
  sorry

end NUMINAMATH_CALUDE_angle_I_measures_138_l168_16800


namespace NUMINAMATH_CALUDE_car_speed_before_servicing_l168_16814

/-- The speed of a car before and after servicing -/
theorem car_speed_before_servicing (speed_serviced : ℝ) (time_serviced time_not_serviced : ℝ) 
  (h1 : speed_serviced = 90)
  (h2 : time_serviced = 3)
  (h3 : time_not_serviced = 6)
  (h4 : speed_serviced * time_serviced = speed_not_serviced * time_not_serviced) :
  speed_not_serviced = 45 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_before_servicing_l168_16814


namespace NUMINAMATH_CALUDE_cube_root_monotone_l168_16858

theorem cube_root_monotone {a b : ℝ} (h : a > b) : a^(1/3) > b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_monotone_l168_16858


namespace NUMINAMATH_CALUDE_f_of_three_equals_six_l168_16812

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(3) = 6 -/
theorem f_of_three_equals_six (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : f 1 = 4)
  (h2 : f 2 = 9)
  (h3 : ∀ x, f x = a * x + b * x + 3) :
  f 3 = 6 := by
sorry

end NUMINAMATH_CALUDE_f_of_three_equals_six_l168_16812


namespace NUMINAMATH_CALUDE_blueberry_picking_l168_16827

theorem blueberry_picking (annie kathryn ben : ℕ) : 
  annie = 8 →
  kathryn = annie + 2 →
  ben = kathryn - 3 →
  annie + kathryn + ben = 25 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_picking_l168_16827


namespace NUMINAMATH_CALUDE_average_not_equal_given_l168_16828

theorem average_not_equal_given (numbers : List ℝ) (given_average : ℝ) : 
  numbers = [12, 13, 14, 510, 520, 530, 1115, 1, 1252140, 2345] →
  given_average = 858.5454545454545 →
  (numbers.sum / numbers.length : ℝ) ≠ given_average := by
sorry

end NUMINAMATH_CALUDE_average_not_equal_given_l168_16828


namespace NUMINAMATH_CALUDE_rod_cutting_l168_16823

theorem rod_cutting (rod_length : Real) (piece_length : Real) :
  rod_length = 42.5 →
  piece_length = 0.85 →
  Int.floor (rod_length / piece_length) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l168_16823


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l168_16825

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 145 → ∃ (a b : ℕ), a^2 - b^2 = 145 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 433 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l168_16825


namespace NUMINAMATH_CALUDE_empty_subset_of_A_l168_16881

def A : Set ℝ := {x : ℝ | x^2 - x = 0}

theorem empty_subset_of_A : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_subset_of_A_l168_16881


namespace NUMINAMATH_CALUDE_bob_candies_count_l168_16809

-- Define Bob's items
def bob_chewing_gums : ℕ := 15
def bob_chocolate_bars : ℕ := 20
def bob_assorted_candies : ℕ := 15

-- Theorem to prove
theorem bob_candies_count : bob_assorted_candies = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_candies_count_l168_16809


namespace NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_perpendicular_parallel_planes_l168_16861

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_perpendicular
  (α : Plane) (m n : Line)
  (h1 : parallel_line_plane m α)
  (h2 : perpendicular_line_plane n α) :
  perpendicular_line_line m n :=
sorry

-- Theorem 2
theorem perpendicular_parallel_planes
  (α β : Plane) (m : Line)
  (h1 : perpendicular_line_plane m α)
  (h2 : parallel_plane_plane α β) :
  perpendicular_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_perpendicular_parallel_planes_l168_16861


namespace NUMINAMATH_CALUDE_rain_probability_l168_16838

theorem rain_probability (p_rain : ℝ) (p_consecutive : ℝ) 
  (h1 : p_rain = 1/3)
  (h2 : p_consecutive = 1/5) :
  p_consecutive / p_rain = 3/5 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_l168_16838


namespace NUMINAMATH_CALUDE_find_x_l168_16844

theorem find_x : ∃ x : ℚ, (3 * x - 6 + 4) / 7 = 15 ∧ x = 107 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l168_16844


namespace NUMINAMATH_CALUDE_u_difference_divisible_l168_16886

/-- Sequence u defined recursively -/
def u (a : ℕ+) : ℕ → ℕ
  | 0 => 1
  | n + 1 => a.val ^ u a n

/-- Theorem stating that n! divides u_{n+1} - u_n for all n ≥ 1 -/
theorem u_difference_divisible (a : ℕ+) (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, u a (n + 1) - u a n = k * n! :=
sorry

end NUMINAMATH_CALUDE_u_difference_divisible_l168_16886


namespace NUMINAMATH_CALUDE_number_equation_solution_l168_16885

theorem number_equation_solution : 
  ∃ (number : ℝ), 35 - (23 - (number - 32)) = 12 * 2 / (1 / 2) ∧ number = 68 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l168_16885


namespace NUMINAMATH_CALUDE_dessert_percentage_l168_16837

/-- Proves that the dessert cost is 25% of the second course price --/
theorem dessert_percentage (initial_amount : ℝ) (first_course_cost : ℝ) 
  (second_course_cost : ℝ) (remaining_amount : ℝ) : ℝ :=
by
  have h1 : initial_amount = 60 := by sorry
  have h2 : first_course_cost = 15 := by sorry
  have h3 : second_course_cost = first_course_cost + 5 := by sorry
  have h4 : remaining_amount = 20 := by sorry

  -- Calculate total spent
  let total_spent := initial_amount - remaining_amount

  -- Calculate dessert cost
  let dessert_cost := total_spent - (first_course_cost + second_course_cost)

  -- Calculate percentage
  let percentage := (dessert_cost / second_course_cost) * 100

  exact 25

end NUMINAMATH_CALUDE_dessert_percentage_l168_16837


namespace NUMINAMATH_CALUDE_family_probability_l168_16872

theorem family_probability (p_boy p_girl : ℝ) (h1 : p_boy = 1 / 2) (h2 : p_girl = 1 / 2) :
  let p_at_least_one_each := 1 - (p_boy ^ 4 + p_girl ^ 4)
  p_at_least_one_each = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_family_probability_l168_16872


namespace NUMINAMATH_CALUDE_sugar_flour_ratio_l168_16893

theorem sugar_flour_ratio (flour baking_soda sugar : ℕ) : 
  (flour = 10 * baking_soda) →
  (flour = 8 * (baking_soda + 60)) →
  (sugar = 2000) →
  (sugar * 6 = flour * 5) :=
by sorry

end NUMINAMATH_CALUDE_sugar_flour_ratio_l168_16893


namespace NUMINAMATH_CALUDE_triathlon_bicycle_speed_specific_triathlon_problem_l168_16862

/-- Triathlon problem -/
theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ) 
  (run_distance run_speed : ℝ) 
  (bike_distance : ℝ) : ℝ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

/-- The specific triathlon problem -/
theorem specific_triathlon_problem : 
  triathlon_bicycle_speed 1.75 (1/3) 1.5 2.5 8 12 = 1728/175 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bicycle_speed_specific_triathlon_problem_l168_16862


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l168_16873

noncomputable def nested_sqrt_sequence : ℕ → ℝ
  | 0 => Real.sqrt 86
  | n + 1 => Real.sqrt (86 + 41 * nested_sqrt_sequence n)

theorem nested_sqrt_value :
  ∃ (limit : ℝ), limit = Real.sqrt (86 + 41 * limit) ∧ limit = 43 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l168_16873


namespace NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l168_16880

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l168_16880


namespace NUMINAMATH_CALUDE_max_value_3m_4n_l168_16818

theorem max_value_3m_4n (m n : ℕ) (h_sum : m * (m + 1) + n^2 ≤ 1987) (h_n_odd : Odd n) :
  3 * m + 4 * n ≤ 221 :=
sorry

end NUMINAMATH_CALUDE_max_value_3m_4n_l168_16818


namespace NUMINAMATH_CALUDE_robot_models_properties_l168_16889

/-- Represents the cost and quantity information for robot models --/
structure RobotModels where
  cost_A : ℕ  -- Cost of model A in yuan
  cost_B : ℕ  -- Cost of model B in yuan
  total_A : ℕ  -- Total spent on model A in yuan
  total_B : ℕ  -- Total spent on model B in yuan
  total_units : ℕ  -- Total units to be purchased

/-- Calculates the maximum number of model A units that can be purchased --/
def max_model_A (r : RobotModels) : ℕ :=
  min ((2 * r.total_units) / 3) r.total_units

/-- Theorem stating the properties of the robot models --/
theorem robot_models_properties (r : RobotModels) 
  (h1 : r.cost_B = 2 * r.cost_A - 400)
  (h2 : r.total_A = 96000)
  (h3 : r.total_B = 168000)
  (h4 : r.total_units = 100) :
  r.cost_A = 1600 ∧ r.cost_B = 2800 ∧ max_model_A r = 66 := by
  sorry

#eval max_model_A ⟨1600, 2800, 96000, 168000, 100⟩

end NUMINAMATH_CALUDE_robot_models_properties_l168_16889


namespace NUMINAMATH_CALUDE_tea_sale_prices_l168_16866

/-- Calculates the sale price per kg for a given tea type -/
def salePricePerKg (quantity : ℕ) (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  (quantity * costPrice + quantity * costPrice * profitPercentage) / quantity

theorem tea_sale_prices :
  let teaA := salePricePerKg 80 15 (25/100)
  let teaB := salePricePerKg 20 20 (30/100)
  let teaC := salePricePerKg 50 25 (20/100)
  let teaD := salePricePerKg 30 30 (15/100)
  teaA = 75/4 ∧ teaB = 26 ∧ teaC = 30 ∧ teaD = 69/2 :=
by sorry

end NUMINAMATH_CALUDE_tea_sale_prices_l168_16866


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l168_16829

-- Define the universe set U
def U : Set Nat := {2, 3, 6, 8}

-- Define set A
def A : Set Nat := {2, 3}

-- Define set B
def B : Set Nat := {2, 6, 8}

-- Theorem statement
theorem complement_intersection_theorem :
  (Aᶜ ∩ B) = {6, 8} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l168_16829


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l168_16857

/-- Given a rectangular plot with the following properties:
  - The length is 20 meters more than the breadth
  - The cost of fencing is 26.50 per meter
  - The total cost of fencing is 5300
  Then the length of the plot is 60 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) : 
  length = breadth + 20 →
  2 * (length + breadth) * 26.5 = 5300 →
  length = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l168_16857


namespace NUMINAMATH_CALUDE_f_properties_l168_16834

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / (sin x + 2)

theorem f_properties (a : ℝ) (h : a ≥ -2) :
  (∀ x ∈ Set.Icc 0 (π/2), Monotone (f π)) ∧
  (∀ x ∈ Set.Icc 0 (π/2), f a x ≤ π/6 - a/3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l168_16834


namespace NUMINAMATH_CALUDE_distance_A_to_C_l168_16821

/-- Proves the distance between city A and C given travel times, distance A to B, and speed ratio -/
theorem distance_A_to_C 
  (eddy_time : ℝ) 
  (freddy_time : ℝ) 
  (distance_AB : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : eddy_time = 3) 
  (h2 : freddy_time = 4) 
  (h3 : distance_AB = 540) 
  (h4 : speed_ratio = 2.4) : 
  distance_AB * freddy_time / (eddy_time * speed_ratio) = 300 := by
sorry

end NUMINAMATH_CALUDE_distance_A_to_C_l168_16821


namespace NUMINAMATH_CALUDE_multiple_with_all_digits_l168_16896

/-- For any integer n, there exists a multiple m of n whose decimal representation
    contains each digit from 0 to 9 at least once. -/
theorem multiple_with_all_digits (n : ℤ) : ∃ m : ℤ,
  (n ∣ m) ∧ (∀ d : ℕ, d < 10 → ∃ k : ℕ, (m.natAbs / 10^k) % 10 = d) := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_all_digits_l168_16896


namespace NUMINAMATH_CALUDE_geometric_progression_sum_l168_16884

/-- A sequence is a geometric progression if the ratio between consecutive terms is constant. -/
def IsGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_progression_sum (a : ℕ → ℝ) :
  IsGeometricProgression a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_l168_16884


namespace NUMINAMATH_CALUDE_union_equality_intersection_equality_l168_16865

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 4 ≤ x ∧ x ≤ 3*m + 2}

-- Theorem for the first question
theorem union_equality (m : ℝ) : A ∪ B m = B m ↔ m ∈ Set.Icc 1 2 := by sorry

-- Theorem for the second question
theorem intersection_equality (m : ℝ) : A ∩ B m = B m ↔ m < -3 := by sorry

end NUMINAMATH_CALUDE_union_equality_intersection_equality_l168_16865


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l168_16815

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l168_16815


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l168_16831

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X : Polynomial ℚ)^5 - 3*(X^3) + X^2 + 2 = 
  (X^2 - 4*X + 6) * q + (-22*X - 28) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l168_16831


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l168_16806

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l168_16806


namespace NUMINAMATH_CALUDE_commemorative_book_sales_l168_16878

/-- Profit function for commemorative book sales -/
def profit (x : ℝ) : ℝ := (x - 20) * (-2 * x + 80)

/-- Theorem for commemorative book sales problem -/
theorem commemorative_book_sales 
  (x : ℝ) 
  (h1 : 20 ≤ x ∧ x ≤ 28) : 
  (∃ (x : ℝ), profit x = 150 ∧ x = 25) ∧ 
  (∀ (y : ℝ), 20 ≤ y ∧ y ≤ 28 → profit y ≤ profit 28) ∧
  profit 28 = 192 := by
  sorry


end NUMINAMATH_CALUDE_commemorative_book_sales_l168_16878


namespace NUMINAMATH_CALUDE_ways_1800_eq_partitions_300_l168_16894

/-- The number of ways to write a positive integer as a sum of ones, twos, and threes, ignoring order -/
def numWays (n : ℕ) : ℕ := sorry

/-- The number of ways to partition a positive integer into four non-negative integer parts -/
def numPartitions4 (n : ℕ) : ℕ := sorry

/-- Theorem stating the equivalence between the two counting problems for n = 1800 -/
theorem ways_1800_eq_partitions_300 : numWays 1800 = numPartitions4 300 := by sorry

end NUMINAMATH_CALUDE_ways_1800_eq_partitions_300_l168_16894


namespace NUMINAMATH_CALUDE_probability_two_cards_sum_15_l168_16883

-- Define the deck
def standard_deck : ℕ := 52

-- Define the number of cards for each value from 2 to 10
def number_cards_per_value : ℕ := 4

-- Define the possible first card values that can sum to 15
def first_card_values : List ℕ := [6, 7, 8, 9, 10]

-- Define the function to calculate the number of ways to choose 2 cards
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

-- State the theorem
theorem probability_two_cards_sum_15 :
  (10 : ℚ) / 331 = (
    (List.sum (first_card_values.map (λ x => 
      if x = 10 then
        number_cards_per_value * number_cards_per_value
      else
        number_cards_per_value * number_cards_per_value
    ))) / (2 * choose_two standard_deck)
  ) := by sorry

end NUMINAMATH_CALUDE_probability_two_cards_sum_15_l168_16883


namespace NUMINAMATH_CALUDE_wrappers_minus_caps_difference_l168_16898

/-- Represents the number of bottle caps Danny found at the park. -/
def bottle_caps_found : ℕ := 11

/-- Represents the number of wrappers Danny found at the park. -/
def wrappers_found : ℕ := 28

/-- Represents the total number of bottle caps in Danny's collection. -/
def total_bottle_caps : ℕ := 68

/-- Represents the total number of wrappers in Danny's collection. -/
def total_wrappers : ℕ := 51

/-- Theorem stating the difference between wrappers and bottle caps found at the park. -/
theorem wrappers_minus_caps_difference : wrappers_found - bottle_caps_found = 17 := by
  sorry

end NUMINAMATH_CALUDE_wrappers_minus_caps_difference_l168_16898
