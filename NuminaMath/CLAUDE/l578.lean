import Mathlib

namespace NUMINAMATH_CALUDE_find_number_l578_57889

theorem find_number : ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 := by sorry

end NUMINAMATH_CALUDE_find_number_l578_57889


namespace NUMINAMATH_CALUDE_product_of_solutions_l578_57886

theorem product_of_solutions : ∃ (x y : ℝ), 
  (abs x = 3 * (abs x - 2)) ∧ 
  (abs y = 3 * (abs y - 2)) ∧ 
  (x ≠ y) ∧ 
  (x * y = -9) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l578_57886


namespace NUMINAMATH_CALUDE_pentagon_lcm_problem_l578_57840

/-- Given five distinct natural numbers on the vertices of a pentagon,
    if the LCM of each pair of adjacent numbers is the same for all sides,
    then the smallest possible value for this common LCM is 30. -/
theorem pentagon_lcm_problem (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  ∃ L : ℕ, L > 0 ∧
    Nat.lcm a b = L ∧
    Nat.lcm b c = L ∧
    Nat.lcm c d = L ∧
    Nat.lcm d e = L ∧
    Nat.lcm e a = L →
  (∀ M : ℕ, M > 0 ∧
    (∃ x y z w v : ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ z ≠ w ∧ z ≠ v ∧ w ≠ v ∧
      Nat.lcm x y = M ∧
      Nat.lcm y z = M ∧
      Nat.lcm z w = M ∧
      Nat.lcm w v = M ∧
      Nat.lcm v x = M) →
    M ≥ 30) :=
by sorry

end NUMINAMATH_CALUDE_pentagon_lcm_problem_l578_57840


namespace NUMINAMATH_CALUDE_limit_sequence_equals_e_to_four_thirds_l578_57848

open Real

theorem limit_sequence_equals_e_to_four_thirds :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((3 * n + 1) / (3 * n - 1)) ^ (2 * n + 3) - Real.exp (4 / 3)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_sequence_equals_e_to_four_thirds_l578_57848


namespace NUMINAMATH_CALUDE_sphere_surface_area_l578_57859

theorem sphere_surface_area (triangle_side_length : ℝ) (center_to_plane_distance : ℝ) : 
  triangle_side_length = 3 →
  center_to_plane_distance = Real.sqrt 7 →
  ∃ (sphere_radius : ℝ),
    sphere_radius ^ 2 = triangle_side_length ^ 2 / 3 + center_to_plane_distance ^ 2 ∧
    4 * Real.pi * sphere_radius ^ 2 = 40 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l578_57859


namespace NUMINAMATH_CALUDE_max_piles_is_30_l578_57892

/-- Represents a configuration of stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : (piles.sum = 660)
  size_constraint : ∀ i j, i < piles.length → j < piles.length → 2 * piles[i]! > piles[j]!

/-- Represents a valid split operation on stone piles -/
def split (sp : StonePiles) (index : Nat) (amount : Nat) : Option StonePiles :=
  sorry

/-- The maximum number of piles that can be formed -/
def max_piles : Nat := 30

/-- Theorem stating that the maximum number of piles is 30 -/
theorem max_piles_is_30 :
  ∀ sp : StonePiles,
  (∀ index amount, split sp index amount = none) →
  sp.piles.length ≤ max_piles :=
sorry

end NUMINAMATH_CALUDE_max_piles_is_30_l578_57892


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l578_57843

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l578_57843


namespace NUMINAMATH_CALUDE_distinct_real_pairs_l578_57849

theorem distinct_real_pairs (x y : ℝ) (hxy : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧
  x^200 - y^200 = 2^199 * (x - y) →
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_distinct_real_pairs_l578_57849


namespace NUMINAMATH_CALUDE_allan_initial_balloons_l578_57825

/-- The number of balloons Allan initially brought to the park -/
def initial_balloons : ℕ := sorry

/-- The number of balloons Allan bought at the park -/
def bought_balloons : ℕ := 3

/-- The total number of balloons Allan had after buying more -/
def total_balloons : ℕ := 8

/-- Theorem stating that Allan initially brought 5 balloons to the park -/
theorem allan_initial_balloons : 
  initial_balloons = total_balloons - bought_balloons := by sorry

end NUMINAMATH_CALUDE_allan_initial_balloons_l578_57825


namespace NUMINAMATH_CALUDE_only_zero_function_satisfies_l578_57850

/-- A function satisfying the given inequality for all non-zero real x and all real y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x ≠ 0 → f (x^2 + y) ≥ (1/x + 1) * f y

/-- The main theorem stating that the only function satisfying the inequality is the zero function -/
theorem only_zero_function_satisfies :
  ∀ f : ℝ → ℝ, SatisfiesInequality f ↔ ∀ x, f x = 0 := by sorry

end NUMINAMATH_CALUDE_only_zero_function_satisfies_l578_57850


namespace NUMINAMATH_CALUDE_triangle_area_l578_57831

/-- The area of a triangle with base 12 and height 5 is 30 -/
theorem triangle_area : 
  let base : ℝ := 12
  let height : ℝ := 5
  (1/2 : ℝ) * base * height = 30 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l578_57831


namespace NUMINAMATH_CALUDE_usb_cost_problem_l578_57861

/-- Given that three identical USBs cost $45, prove that seven such USBs cost $105. -/
theorem usb_cost_problem (cost_of_three : ℝ) (h1 : cost_of_three = 45) : 
  (7 / 3) * cost_of_three = 105 := by
  sorry

end NUMINAMATH_CALUDE_usb_cost_problem_l578_57861


namespace NUMINAMATH_CALUDE_log_sum_property_l578_57832

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_property_l578_57832


namespace NUMINAMATH_CALUDE_train_delay_l578_57835

/-- Proves that a train moving at 6/7 of its usual speed will be 30 minutes late on a journey that usually takes 3 hours. -/
theorem train_delay (usual_speed : ℝ) (usual_time : ℝ) (h1 : usual_time = 3) :
  let current_speed := (6/7) * usual_speed
  let current_time := usual_speed * usual_time / current_speed
  (current_time - usual_time) * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_delay_l578_57835


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l578_57800

theorem mean_of_three_numbers (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 92 →
  d = 120 →
  b = 60 →
  (a + b + c) / 3 = 82 + 2/3 :=
by sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l578_57800


namespace NUMINAMATH_CALUDE_manager_selection_l578_57883

theorem manager_selection (n m k : ℕ) (h1 : n = 8) (h2 : m = 4) (h3 : k = 2) : 
  (n.choose m) - ((n - k).choose (m - k)) = 55 := by
  sorry

end NUMINAMATH_CALUDE_manager_selection_l578_57883


namespace NUMINAMATH_CALUDE_parity_of_f_l578_57865

/-- A function that is not always zero -/
def NonZeroFunction (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

/-- Definition of an odd function -/
def OddFunction (F : ℝ → ℝ) : Prop :=
  ∀ x, F (-x) = -F x

/-- Definition of an even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The main theorem -/
theorem parity_of_f (f : ℝ → ℝ) (h_nonzero : NonZeroFunction f) :
    let F := fun x => if x ≠ 0 then (x^3 - 2*x) * f x else 0
    OddFunction F → EvenFunction f := by
  sorry

end NUMINAMATH_CALUDE_parity_of_f_l578_57865


namespace NUMINAMATH_CALUDE_stock_price_decrease_l578_57824

theorem stock_price_decrease (F : ℝ) (h1 : F > 0) : 
  let J := 0.9 * F
  let M := 0.8 * J
  (F - M) / F * 100 = 28 := by
sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l578_57824


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l578_57838

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (al_count : ℕ) (o_count : ℕ) (h_count : ℕ) 
  (al_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) : ℝ :=
  al_count * al_weight + o_count * o_weight + h_count * h_weight

/-- The molecular weight of the compound AlO₃H₃ is 78.01 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 3 3 26.98 16.00 1.01 = 78.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l578_57838


namespace NUMINAMATH_CALUDE_inconsistent_division_problem_l578_57879

theorem inconsistent_division_problem 
  (x y q : ℕ+) 
  (h1 : x = 9 * y + 4)
  (h2 : 2 * x = 7 * q + 1)
  (h3 : 5 * y - x = 3) :
  False :=
sorry

end NUMINAMATH_CALUDE_inconsistent_division_problem_l578_57879


namespace NUMINAMATH_CALUDE_number_equation_solution_l578_57808

theorem number_equation_solution :
  ∃ (x : ℝ), 7 * x = 3 * x + 12 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l578_57808


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l578_57887

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there are 9 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l578_57887


namespace NUMINAMATH_CALUDE_hall_volume_l578_57844

/-- A rectangular hall with specific dimensions and area properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  area_equality : 2 * (length * width) = 2 * (length * height + width * height)

/-- The volume of a rectangular hall is 900 cubic meters -/
theorem hall_volume (hall : RectangularHall) 
  (h_length : hall.length = 15)
  (h_width : hall.width = 10) : 
  hall.length * hall.width * hall.height = 900 := by
  sorry

#check hall_volume

end NUMINAMATH_CALUDE_hall_volume_l578_57844


namespace NUMINAMATH_CALUDE_tina_fruit_difference_l578_57806

/-- Calculates the difference between remaining tangerines and oranges in Tina's bag --/
def remaining_difference (initial_oranges initial_tangerines removed_oranges removed_tangerines : ℕ) : ℕ :=
  (initial_tangerines - removed_tangerines) - (initial_oranges - removed_oranges)

/-- Proves that the difference between remaining tangerines and oranges is 4 --/
theorem tina_fruit_difference :
  remaining_difference 5 17 2 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tina_fruit_difference_l578_57806


namespace NUMINAMATH_CALUDE_min_value_expression_l578_57870

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 8*a*b + 24*b^2 + 16*b*c + 6*c^2 ≥ 18 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1 ∧
    a₀^2 + 8*a₀*b₀ + 24*b₀^2 + 16*b₀*c₀ + 6*c₀^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l578_57870


namespace NUMINAMATH_CALUDE_remaining_amount_is_correct_l578_57815

-- Define the problem parameters
def initial_amount : ℚ := 100
def action_figure_quantity : ℕ := 3
def board_game_quantity : ℕ := 2
def puzzle_set_quantity : ℕ := 4
def action_figure_price : ℚ := 12
def board_game_price : ℚ := 11
def puzzle_set_price : ℚ := 6
def action_figure_discount : ℚ := 0.25
def sales_tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining amount
def calculate_remaining_amount : ℚ :=
  let discounted_action_figure_price := action_figure_price * (1 - action_figure_discount)
  let action_figure_total := discounted_action_figure_price * action_figure_quantity
  let board_game_total := board_game_price * board_game_quantity
  let puzzle_set_total := puzzle_set_price * puzzle_set_quantity
  let subtotal := action_figure_total + board_game_total + puzzle_set_total
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  initial_amount - total_with_tax

-- Theorem statement
theorem remaining_amount_is_correct :
  calculate_remaining_amount = 23.35 := by sorry

end NUMINAMATH_CALUDE_remaining_amount_is_correct_l578_57815


namespace NUMINAMATH_CALUDE_total_rope_length_l578_57896

def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss : ℝ := 1.2

theorem total_rope_length :
  let initial_length := rope_lengths.sum
  let num_knots := rope_lengths.length - 1
  let total_loss := num_knots * knot_loss
  initial_length - total_loss = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_rope_length_l578_57896


namespace NUMINAMATH_CALUDE_quadratic_factorization_l578_57823

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 15 * y^2 - 82 * y + 56 = (C * y - 14) * (D * y - 4)) →
  C * D + C = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l578_57823


namespace NUMINAMATH_CALUDE_yellow_green_difference_l578_57841

/-- The number of buttons purchased by a tailor -/
def total_buttons : ℕ := 275

/-- The number of green buttons purchased -/
def green_buttons : ℕ := 90

/-- The number of blue buttons purchased -/
def blue_buttons : ℕ := green_buttons - 5

/-- The number of yellow buttons purchased -/
def yellow_buttons : ℕ := total_buttons - green_buttons - blue_buttons

/-- Theorem stating the difference between yellow and green buttons -/
theorem yellow_green_difference : 
  yellow_buttons - green_buttons = 10 := by sorry

end NUMINAMATH_CALUDE_yellow_green_difference_l578_57841


namespace NUMINAMATH_CALUDE_max_player_salary_l578_57876

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 15 →
  min_salary = 20000 →
  max_total = 800000 →
  (∃ (salaries : Fin n → ℝ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ max_total) ∧
    (∀ i, salaries i ≤ 520000)) ∧
  ¬(∃ (salaries : Fin n → ℝ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ max_total) ∧
    (∃ i, salaries i > 520000)) :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l578_57876


namespace NUMINAMATH_CALUDE_village_population_after_events_l578_57874

theorem village_population_after_events (initial_population : ℕ) : 
  initial_population = 7800 → 
  (initial_population - initial_population / 10 - 
   (initial_population - initial_population / 10) / 4) = 5265 := by
sorry

end NUMINAMATH_CALUDE_village_population_after_events_l578_57874


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l578_57873

theorem grocery_store_inventory (ordered : ℕ) (sold : ℕ) (storeroom : ℕ) 
  (h1 : ordered = 4458)
  (h2 : sold = 1561)
  (h3 : storeroom = 575) :
  ordered - sold + storeroom = 3472 :=
by sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l578_57873


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_order_l578_57854

open Complex

theorem smallest_root_of_unity_order : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^5 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_order_l578_57854


namespace NUMINAMATH_CALUDE_roots_imply_k_range_l578_57856

/-- The quadratic function f(x) = 2x^2 - kx + k - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + k - 3

theorem roots_imply_k_range :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, 
    f k x₁ = 0 ∧ 0 < x₁ ∧ x₁ < 1 ∧
    f k x₂ = 0 ∧ 1 < x₂ ∧ x₂ < 2) →
  3 < k ∧ k < 5 :=
by sorry

end NUMINAMATH_CALUDE_roots_imply_k_range_l578_57856


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l578_57869

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 - 64 * x - 4 * y^2 + 8 * y + 60 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance (h : ∃ x y, hyperbola_equation x y) : ℝ :=
  1

theorem hyperbola_vertex_distance :
  ∀ h : ∃ x y, hyperbola_equation x y,
  vertex_distance h = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l578_57869


namespace NUMINAMATH_CALUDE_complement_union_theorem_l578_57846

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_union_theorem :
  (A ∪ B)ᶜ = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l578_57846


namespace NUMINAMATH_CALUDE_power_nine_2023_mod_50_l578_57890

theorem power_nine_2023_mod_50 : 9^2023 % 50 = 29 := by
  sorry

end NUMINAMATH_CALUDE_power_nine_2023_mod_50_l578_57890


namespace NUMINAMATH_CALUDE_mn_positive_necessary_mn_positive_not_sufficient_l578_57813

/-- Definition of an ellipse equation -/
def is_ellipse_equation (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1 ∧ m ≠ n ∧ m > 0 ∧ n > 0

/-- The condition mn > 0 is necessary for the equation to represent an ellipse -/
theorem mn_positive_necessary (m n : ℝ) :
  is_ellipse_equation m n → m * n > 0 :=
sorry

/-- The condition mn > 0 is not sufficient for the equation to represent an ellipse -/
theorem mn_positive_not_sufficient :
  ∃ (m n : ℝ), m * n > 0 ∧ ¬(is_ellipse_equation m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_mn_positive_not_sufficient_l578_57813


namespace NUMINAMATH_CALUDE_statement_is_false_l578_57872

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem statement_is_false : ∃ n : ℕ, 
  (sum_of_digits n % 6 = 0) ∧ (n % 6 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_statement_is_false_l578_57872


namespace NUMINAMATH_CALUDE_sum_and_square_difference_l578_57857

theorem sum_and_square_difference (x y : ℝ) : 
  x + y = 15 → x^2 - y^2 = 150 → x - y = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_and_square_difference_l578_57857


namespace NUMINAMATH_CALUDE_milburg_grown_ups_l578_57821

/-- The population of Milburg -/
def total_population : ℕ := 8243

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := total_population - children

/-- Theorem stating that the number of grown-ups in Milburg is 5256 -/
theorem milburg_grown_ups : grown_ups = 5256 := by
  sorry

end NUMINAMATH_CALUDE_milburg_grown_ups_l578_57821


namespace NUMINAMATH_CALUDE_soccer_ball_inflation_l578_57880

/-- Proves that Ermias inflated 5 more balls than Alexia given the problem conditions -/
theorem soccer_ball_inflation (inflation_time ball_count_alexia total_time : ℕ) 
  (h1 : inflation_time = 20)
  (h2 : ball_count_alexia = 20)
  (h3 : total_time = 900) : 
  ∃ (additional_balls : ℕ), 
    inflation_time * ball_count_alexia + 
    inflation_time * (ball_count_alexia + additional_balls) = total_time ∧ 
    additional_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_inflation_l578_57880


namespace NUMINAMATH_CALUDE_trigonometric_identity_l578_57826

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (π / 4 + α) = Real.sqrt 2 / 3) : 
  Real.sin (2 * α) / (1 - Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l578_57826


namespace NUMINAMATH_CALUDE_jakes_weight_l578_57885

theorem jakes_weight (jake_weight sister_weight : ℝ) : 
  (0.8 * jake_weight = 2 * sister_weight) →
  (jake_weight + sister_weight = 168) →
  (jake_weight = 120) := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l578_57885


namespace NUMINAMATH_CALUDE_parabola_chord_length_l578_57829

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Theorem statement
theorem parabola_chord_length :
  ∀ p : ℝ,
  p > 0 →
  (∃ x y : ℝ, parabola p x y ∧ x = 1 ∧ y = 0) →  -- Focus at (1, 0)
  (∃ A B : ℝ × ℝ,
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B) →
  (∀ x y : ℝ, parabola p x y ↔ y^2 = 2*x) ∧     -- Standard equation
  (∃ A B : ℝ × ℝ,
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4) -- Chord length
  := by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l578_57829


namespace NUMINAMATH_CALUDE_roots_difference_squared_l578_57837

theorem roots_difference_squared (α β : ℝ) : 
  α^2 - 3*α + 2 = 0 → β^2 - 3*β + 2 = 0 → α ≠ β → (α - β)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_difference_squared_l578_57837


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l578_57830

theorem cubic_root_equation_solutions :
  let f (x : ℝ) := Real.rpow (18 * x - 2) (1/3) + Real.rpow (16 * x + 2) (1/3) - 6 * Real.rpow x (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -1/12 ∨ x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l578_57830


namespace NUMINAMATH_CALUDE_gift_contribution_ratio_l578_57884

theorem gift_contribution_ratio : 
  let lisa_savings : ℚ := 1200
  let mother_contribution : ℚ := 3/5 * lisa_savings
  let total_needed : ℚ := 3760
  let shortfall : ℚ := 400
  let total_contributions : ℚ := total_needed - shortfall
  let brother_contribution : ℚ := total_contributions - lisa_savings - mother_contribution
  brother_contribution / mother_contribution = 2 := by sorry

end NUMINAMATH_CALUDE_gift_contribution_ratio_l578_57884


namespace NUMINAMATH_CALUDE_factor_grid_theorem_l578_57804

/-- The factors of 100 -/
def factors_of_100 : Finset Nat := {1, 2, 4, 5, 10, 20, 25, 50, 100}

/-- The product of all factors of 100 -/
def product_of_factors : Nat := Finset.prod factors_of_100 id

/-- The common product for each row, column, and diagonal -/
def common_product : Nat := 1000

/-- The 3x3 grid representation -/
structure Grid :=
  (a b c d e f g h i : Nat)

/-- Predicate to check if a grid is valid -/
def is_valid_grid (grid : Grid) : Prop :=
  grid.a ∈ factors_of_100 ∧ grid.b ∈ factors_of_100 ∧ grid.c ∈ factors_of_100 ∧
  grid.d ∈ factors_of_100 ∧ grid.e ∈ factors_of_100 ∧ grid.f ∈ factors_of_100 ∧
  grid.g ∈ factors_of_100 ∧ grid.h ∈ factors_of_100 ∧ grid.i ∈ factors_of_100

/-- Predicate to check if a grid satisfies the product condition -/
def satisfies_product_condition (grid : Grid) : Prop :=
  grid.a * grid.b * grid.c = common_product ∧
  grid.d * grid.e * grid.f = common_product ∧
  grid.g * grid.h * grid.i = common_product ∧
  grid.a * grid.d * grid.g = common_product ∧
  grid.b * grid.e * grid.h = common_product ∧
  grid.c * grid.f * grid.i = common_product ∧
  grid.a * grid.e * grid.i = common_product ∧
  grid.c * grid.e * grid.g = common_product

/-- The main theorem -/
theorem factor_grid_theorem (x : Nat) :
  is_valid_grid { a := x, b := 1, c := 50, d := 2, e := 25, f := 20, g := 10, h := 4, i := 5 } ∧
  satisfies_product_condition { a := x, b := 1, c := 50, d := 2, e := 25, f := 20, g := 10, h := 4, i := 5 } →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_factor_grid_theorem_l578_57804


namespace NUMINAMATH_CALUDE_difference_in_cost_l578_57809

def joy_pencils : ℕ := 30
def colleen_pencils : ℕ := 50
def pencil_cost : ℕ := 4

theorem difference_in_cost : (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_cost_l578_57809


namespace NUMINAMATH_CALUDE_cakes_slices_problem_l578_57898

theorem cakes_slices_problem (total_slices : ℕ) (friends_fraction : ℚ) 
  (family_fraction : ℚ) (eaten_slices : ℕ) (remaining_slices : ℕ) :
  total_slices = 16 →
  family_fraction = 1/3 →
  eaten_slices = 3 →
  remaining_slices = 5 →
  (1 - friends_fraction) * (1 - family_fraction) * total_slices - eaten_slices = remaining_slices →
  friends_fraction = 1/4 := by
sorry

end NUMINAMATH_CALUDE_cakes_slices_problem_l578_57898


namespace NUMINAMATH_CALUDE_categorical_variables_l578_57891

-- Define the variables
def Smoking : Type := String
def Gender : Type := String
def ReligiousBelief : Type := String
def Nationality : Type := String

-- Define what it means for a variable to be categorical
def IsCategorical (α : Type) : Prop := ∃ (categories : Set α), Finite categories ∧ (∀ x : α, x ∈ categories)

-- State the theorem
theorem categorical_variables :
  IsCategorical Gender ∧ IsCategorical ReligiousBelief ∧ IsCategorical Nationality :=
sorry

end NUMINAMATH_CALUDE_categorical_variables_l578_57891


namespace NUMINAMATH_CALUDE_race_distance_l578_57853

/-- Represents the race scenario with given conditions -/
structure RaceScenario where
  distance : ℝ
  timeA : ℝ
  startAdvantage1 : ℝ
  timeDifference : ℝ
  startAdvantage2 : ℝ

/-- Defines the conditions of the race -/
def raceConditions : RaceScenario → Prop
  | ⟨d, t, s1, dt, s2⟩ => 
    t = 77.5 ∧ 
    s1 = 25 ∧ 
    dt = 10 ∧ 
    s2 = 45 ∧ 
    d / t = (d - s1) / (t + dt) ∧ 
    d / t = (d - s2) / t

/-- Theorem stating that the race distance is 218.75 meters -/
theorem race_distance (scenario : RaceScenario) 
  (h : raceConditions scenario) : scenario.distance = 218.75 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l578_57853


namespace NUMINAMATH_CALUDE_integer_solutions_system_l578_57810

theorem integer_solutions_system :
  ∀ x y z : ℤ,
  (x + y = 1 - z ∧ x^3 + y^3 = 1 - z^2) ↔
  ((∃ k : ℤ, x = k ∧ y = -k ∧ z = 1) ∨
   (x = 0 ∧ y = 1 ∧ z = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 0) ∨
   (x = 0 ∧ y = -2 ∧ z = 3) ∨
   (x = -2 ∧ y = 0 ∧ z = 3) ∨
   (x = -2 ∧ y = -3 ∧ z = 6) ∨
   (x = -3 ∧ y = -2 ∧ z = 6)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l578_57810


namespace NUMINAMATH_CALUDE_sqrt_defined_iff_l578_57802

theorem sqrt_defined_iff (x : ℝ) : Real.sqrt (5 - 3 * x) ≥ 0 ↔ x ≤ 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_defined_iff_l578_57802


namespace NUMINAMATH_CALUDE_stratified_sampling_l578_57814

/-- Stratified sampling problem -/
theorem stratified_sampling 
  (total_employees : ℕ) 
  (middle_managers : ℕ) 
  (senior_managers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 150) 
  (h2 : middle_managers = 30) 
  (h3 : senior_managers = 10) 
  (h4 : sample_size = 30) :
  (sample_size * middle_managers / total_employees = 6) ∧ 
  (sample_size * senior_managers / total_employees = 2) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l578_57814


namespace NUMINAMATH_CALUDE_students_allowance_l578_57839

theorem students_allowance (allowance : ℚ) : 
  (2 / 3 : ℚ) * (2 / 5 : ℚ) * allowance = 6 / 10 → 
  allowance = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_students_allowance_l578_57839


namespace NUMINAMATH_CALUDE_trig_equation_solution_l578_57811

theorem trig_equation_solution (x : Real) : 
  (6 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.cos x ^ 2 = 2) ↔ 
  (∃ k : Int, x = -π/4 + π * k) ∨ 
  (∃ n : Int, x = Real.arctan (3/4) + π * n) := by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l578_57811


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l578_57827

theorem sum_of_cubes_difference (a b c : ℕ+) :
  (a + b + c : ℕ)^3 - a^3 - b^3 - c^3 = 180 → a + b + c = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l578_57827


namespace NUMINAMATH_CALUDE_arctan_sum_equation_n_unique_l578_57812

/-- The positive integer n satisfying the equation arctan(1/2) + arctan(1/3) + arctan(1/7) + arctan(1/n) = π/4 -/
def n : ℕ := 7

/-- The equation that n satisfies -/
theorem arctan_sum_equation : 
  Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/4 := by
  sorry

/-- Proof that n is the unique positive integer satisfying the equation -/
theorem n_unique : 
  ∀ m : ℕ, m > 0 → 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/m) = π/4) → 
  m = n := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_n_unique_l578_57812


namespace NUMINAMATH_CALUDE_octagon_diagonals_l578_57847

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: An octagon has 20 internal diagonals -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l578_57847


namespace NUMINAMATH_CALUDE_davids_english_marks_l578_57845

def marks_mathematics : ℕ := 85
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 87
def marks_biology : ℕ := 95
def average_marks : ℕ := 89
def number_of_subjects : ℕ := 5

theorem davids_english_marks :
  ∃ (marks_english : ℕ),
    marks_english +
    marks_mathematics +
    marks_physics +
    marks_chemistry +
    marks_biology =
    average_marks * number_of_subjects ∧
    marks_english = 86 := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l578_57845


namespace NUMINAMATH_CALUDE_difference_sum_of_T_l578_57868

def T : Finset ℕ := Finset.range 9

def difference_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if 3^j > 3^i then 3^j - 3^i else 0))

theorem difference_sum_of_T : difference_sum T = 69022 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_T_l578_57868


namespace NUMINAMATH_CALUDE_inequality_system_solution_l578_57833

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l578_57833


namespace NUMINAMATH_CALUDE_probability_a_speaks_truth_l578_57842

theorem probability_a_speaks_truth 
  (prob_b : ℝ)
  (prob_a_and_b : ℝ)
  (h1 : prob_b = 0.60)
  (h2 : prob_a_and_b = 0.51)
  (h3 : ∃ (prob_a : ℝ), prob_a_and_b = prob_a * prob_b) :
  ∃ (prob_a : ℝ), prob_a = 0.85 := by
sorry

end NUMINAMATH_CALUDE_probability_a_speaks_truth_l578_57842


namespace NUMINAMATH_CALUDE_correct_team_selection_l578_57893

def group_A_nurses : ℕ := 4
def group_A_doctors : ℕ := 1
def group_B_nurses : ℕ := 6
def group_B_doctors : ℕ := 2
def members_per_group : ℕ := 2
def total_members : ℕ := 4
def required_doctors : ℕ := 1

def select_team : ℕ := sorry

theorem correct_team_selection :
  select_team = 132 := by sorry

end NUMINAMATH_CALUDE_correct_team_selection_l578_57893


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l578_57871

theorem square_cut_perimeter (square_side : ℝ) (total_perimeter : ℝ) :
  square_side = 4 →
  total_perimeter = 25 →
  ∃ (rect1_length rect1_width rect2_length rect2_width : ℝ),
    rect1_length * rect1_width + rect2_length * rect2_width = square_side * square_side ∧
    2 * (rect1_length + rect1_width) + 2 * (rect2_length + rect2_width) = total_perimeter :=
by sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_l578_57871


namespace NUMINAMATH_CALUDE_water_jar_problem_l578_57899

theorem water_jar_problem (c_s c_l : ℝ) (h1 : c_s > 0) (h2 : c_l > 0) (h3 : c_s ≠ c_l) : 
  (1 / 6 : ℝ) * c_s = (1 / 5 : ℝ) * c_l → 
  (1 / 5 : ℝ) + (1 / 6 : ℝ) * c_s / c_l = (2 / 5 : ℝ) := by
  sorry

#check water_jar_problem

end NUMINAMATH_CALUDE_water_jar_problem_l578_57899


namespace NUMINAMATH_CALUDE_computer_contract_probability_l578_57881

theorem computer_contract_probability (p_hardware : ℚ) (p_not_software : ℚ) (p_at_least_one : ℚ)
  (h1 : p_hardware = 3 / 4)
  (h2 : p_not_software = 3 / 5)
  (h3 : p_at_least_one = 5 / 6) :
  p_hardware + (1 - p_not_software) - p_at_least_one = 19 / 60 :=
by sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l578_57881


namespace NUMINAMATH_CALUDE_tank_capacity_is_120_gallons_l578_57834

/-- Represents the capacity of a water tank in gallons -/
def tank_capacity : ℝ := 120

/-- Represents the difference in gallons between 70% and 40% full -/
def difference : ℝ := 36

/-- Theorem stating that the tank capacity is 120 gallons -/
theorem tank_capacity_is_120_gallons : 
  (0.7 * tank_capacity - 0.4 * tank_capacity = difference) → 
  tank_capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_120_gallons_l578_57834


namespace NUMINAMATH_CALUDE_parabola_directrix_l578_57864

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -17/4

/-- Theorem: The directrix of the given parabola is y = -17/4 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola x y → ∃ (d : ℝ), directrix d ∧ 
  (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
    (p.1 - 4)^2 + (p.2 - d)^2 = (p.2 - (d + 4))^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l578_57864


namespace NUMINAMATH_CALUDE_identifier_count_l578_57807

/-- The number of English letters -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible characters for the second and third positions -/
def num_chars : ℕ := num_letters + num_digits

/-- The total number of possible identifiers -/
def total_identifiers : ℕ := num_letters + (num_letters * num_chars) + (num_letters * num_chars * num_chars)

theorem identifier_count : total_identifiers = 34658 := by
  sorry

end NUMINAMATH_CALUDE_identifier_count_l578_57807


namespace NUMINAMATH_CALUDE_intercept_sum_modulo_40_l578_57828

/-- 
Given the congruence 5x ≡ 3y - 2 (mod 40), this theorem proves that 
the sum of the x-intercept and y-intercept is 38, where both intercepts 
are non-negative integers less than 40.
-/
theorem intercept_sum_modulo_40 : ∃ (x₀ y₀ : ℕ), 
  x₀ < 40 ∧ y₀ < 40 ∧ 
  (5 * x₀) % 40 = (3 * 0 - 2) % 40 ∧
  (5 * 0) % 40 = (3 * y₀ - 2) % 40 ∧
  x₀ + y₀ = 38 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_modulo_40_l578_57828


namespace NUMINAMATH_CALUDE_sum_of_terms_l578_57822

theorem sum_of_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = n^2 + n + 1) →
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) →
  a 8 + a 9 + a 10 + a 11 + a 12 = 100 := by
sorry

end NUMINAMATH_CALUDE_sum_of_terms_l578_57822


namespace NUMINAMATH_CALUDE_acidic_concentration_after_water_removal_l578_57803

/-- Calculates the final concentration of an acidic solution after removing water -/
theorem acidic_concentration_after_water_removal
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_removed : ℝ)
  (h1 : initial_volume = 27)
  (h2 : initial_concentration = 0.4)
  (h3 : water_removed = 9)
  : (initial_volume * initial_concentration) / (initial_volume - water_removed) = 0.6 := by
  sorry

#check acidic_concentration_after_water_removal

end NUMINAMATH_CALUDE_acidic_concentration_after_water_removal_l578_57803


namespace NUMINAMATH_CALUDE_union_when_a_is_two_intersection_empty_iff_l578_57877

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3 ∧ a > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- Theorem 1: When a = 2, A ∪ B = {x | -2 < x < 7}
theorem union_when_a_is_two : 
  A 2 ∪ B = {x : ℝ | -2 < x ∧ x < 7} := by sorry

-- Theorem 2: A ∩ B = ∅ if and only if a ≥ 5
theorem intersection_empty_iff : 
  ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_union_when_a_is_two_intersection_empty_iff_l578_57877


namespace NUMINAMATH_CALUDE_triangle_cosine_identities_l578_57888

theorem triangle_cosine_identities (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  (Real.cos (2 * α) + Real.cos (2 * β) + Real.cos (2 * γ) + 4 * Real.cos α * Real.cos β * Real.cos γ + 1 = 0) ∧
  (Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 + 2 * Real.cos α * Real.cos β * Real.cos γ = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_identities_l578_57888


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l578_57852

theorem triangle_side_lengths (x y z k : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (k_ge_2 : k ≥ 2) 
  (prod_cond : x * y * z ≤ 2) 
  (sum_cond : 1 / x^2 + 1 / y^2 + 1 / z^2 < k) :
  (∃ a b c : ℝ, a = x ∧ b = y ∧ c = z ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (2 ≤ k ∧ k ≤ 9/4) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l578_57852


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l578_57818

/-- The area of a triangle given its side lengths -/
noncomputable def S (a b c : ℝ) : ℝ := sorry

/-- Triangle inequality -/
def is_triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_area_inequality 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : is_triangle a₁ b₁ c₁) 
  (h₂ : is_triangle a₂ b₂ c₂) : 
  Real.sqrt (S a₁ b₁ c₁) + Real.sqrt (S a₂ b₂ c₂) ≤ Real.sqrt (S (a₁ + a₂) (b₁ + b₂) (c₁ + c₂)) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l578_57818


namespace NUMINAMATH_CALUDE_dealership_sales_prediction_l578_57851

/-- Represents the sales ratio of different car types -/
structure SalesRatio where
  sports : ℕ
  sedans : ℕ
  suvs : ℕ

/-- Represents the expected sales of different car types -/
structure ExpectedSales where
  sports : ℕ
  sedans : ℕ
  suvs : ℕ

/-- Given a sales ratio and expected sports car sales, calculates the expected sales of all car types -/
def calculateExpectedSales (ratio : SalesRatio) (expectedSports : ℕ) : ExpectedSales :=
  { sports := expectedSports,
    sedans := expectedSports * ratio.sedans / ratio.sports,
    suvs := expectedSports * ratio.suvs / ratio.sports }

theorem dealership_sales_prediction 
  (ratio : SalesRatio)
  (expectedSports : ℕ)
  (h1 : ratio.sports = 5)
  (h2 : ratio.sedans = 8)
  (h3 : ratio.suvs = 3)
  (h4 : expectedSports = 35) :
  let expected := calculateExpectedSales ratio expectedSports
  expected.sedans = 56 ∧ expected.suvs = 21 := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_prediction_l578_57851


namespace NUMINAMATH_CALUDE_maude_age_l578_57897

theorem maude_age (anne emile maude : ℕ) 
  (h1 : anne = 96)
  (h2 : anne = 2 * emile)
  (h3 : emile = 6 * maude) :
  maude = 8 := by
sorry

end NUMINAMATH_CALUDE_maude_age_l578_57897


namespace NUMINAMATH_CALUDE_probability_one_third_implies_five_l578_57866

def integer_list : List ℕ := [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

def count (n : ℕ) (l : List ℕ) : ℕ := (l.filter (· = n)).length

theorem probability_one_third_implies_five :
  ∀ n : ℕ, 
  (count n integer_list : ℚ) / (integer_list.length : ℚ) = 1 / 3 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_probability_one_third_implies_five_l578_57866


namespace NUMINAMATH_CALUDE_inequality_proof_l578_57801

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (a^3 / (1 + b * c)) + Real.sqrt (b^3 / (1 + a * c)) + Real.sqrt (c^3 / (1 + a * b)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l578_57801


namespace NUMINAMATH_CALUDE_sin_cos_identity_l578_57895

theorem sin_cos_identity : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l578_57895


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l578_57805

/-- A sufficient but not necessary condition for a quadratic function to have no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (ha : a ≠ 0) :
  b^2 - 4*a*c < -1 → ∀ x, a*x^2 + b*x + c ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l578_57805


namespace NUMINAMATH_CALUDE_ball_drawing_game_l578_57820

/-- Represents the probability that the last ball is white in the ball-drawing game. -/
def lastBallWhiteProbability (p : ℕ) : ℚ :=
  if p % 2 = 0 then 0 else 1

/-- The ball-drawing game theorem. -/
theorem ball_drawing_game (p q : ℕ) :
  ∀ (pile : ℕ), lastBallWhiteProbability p = if p % 2 = 0 then 0 else 1 := by
  sorry

#check ball_drawing_game

end NUMINAMATH_CALUDE_ball_drawing_game_l578_57820


namespace NUMINAMATH_CALUDE_x_plus_y_value_l578_57875

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 18/5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l578_57875


namespace NUMINAMATH_CALUDE_sum_S_six_cards_l578_57882

/-- The number of strictly increasing subsequences of length 2 or more in a sequence -/
def S (π : List ℕ) : ℕ := sorry

/-- The sum of S(π) over all permutations of n distinct elements -/
def sum_S (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of S(π) over all permutations of 6 distinct elements is 8287 -/
theorem sum_S_six_cards : sum_S 6 = 8287 := by sorry

end NUMINAMATH_CALUDE_sum_S_six_cards_l578_57882


namespace NUMINAMATH_CALUDE_log_sum_lower_bound_l578_57819

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_lower_bound :
  (∀ a : ℝ, a > 1 → log 2 a + log a 2 ≥ 2) ∧
  (∃ m : ℝ, m < 2 ∧ ∀ a : ℝ, a > 1 → log 2 a + log a 2 ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_log_sum_lower_bound_l578_57819


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l578_57817

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_max_sum :
  let a₁ : ℚ := 4
  let d : ℚ := -5/7
  ∀ n : ℕ, n ≠ 0 → arithmeticSum a₁ d 6 ≥ arithmeticSum a₁ d n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l578_57817


namespace NUMINAMATH_CALUDE_sum_a_c_equals_six_l578_57863

theorem sum_a_c_equals_six (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 36) 
  (h2 : b + d = 6) : 
  a + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_a_c_equals_six_l578_57863


namespace NUMINAMATH_CALUDE_parabola_midpoint_distance_l578_57858

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 + 3*x + 2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

-- Define the square of the distance between two points
def square_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Theorem statement
theorem parabola_midpoint_distance 
  (C D : PointOnParabola) 
  (h : is_midpoint C.x C.y D.x D.y) : 
  square_distance C.x C.y D.x D.y = 16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_distance_l578_57858


namespace NUMINAMATH_CALUDE_probability_of_sum_15_l578_57860

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face

/-- A standard 52-card deck --/
def Deck : Finset Card :=
  sorry

/-- Predicate for a card being a number card (2 through 10) --/
def isNumberCard (c : Card) : Prop :=
  match c with
  | Card.Number n => 2 ≤ n ∧ n ≤ 10
  | Card.Face => False

/-- Predicate for two cards summing to 15 --/
def sumsTo15 (c1 c2 : Card) : Prop :=
  match c1, c2 with
  | Card.Number n1, Card.Number n2 => n1 + n2 = 15
  | _, _ => False

/-- The probability of selecting two number cards that sum to 15 --/
def probabilityOfSum15 : ℚ :=
  sorry

theorem probability_of_sum_15 :
  probabilityOfSum15 = 8 / 442 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_15_l578_57860


namespace NUMINAMATH_CALUDE_fraction_equality_l578_57862

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) 
  (h2 : (2 * a) / (3 * b) + (a + 12 * b) / (3 * b + 12 * a) = 5 / 3) : 
  a / b = -93 / 49 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l578_57862


namespace NUMINAMATH_CALUDE_misread_subtraction_l578_57894

theorem misread_subtraction (x y : Nat) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y = 9 →  -- Two-digit number condition
  10 * x + 6 - 57 = 39 →  -- Misread calculation result
  10 * x + y - 57 = 42    -- Correct calculation result
:= by sorry

end NUMINAMATH_CALUDE_misread_subtraction_l578_57894


namespace NUMINAMATH_CALUDE_hexagon_division_l578_57816

/-- A regular hexagon with all sides and diagonals drawn -/
structure RegularHexagonWithDiagonals where
  /-- The number of vertices in a regular hexagon -/
  num_vertices : Nat
  /-- The number of sides in a regular hexagon -/
  num_sides : Nat
  /-- The number of diagonals in a regular hexagon -/
  num_diagonals : Nat
  /-- Assertion that the number of vertices is 6 -/
  vertex_count : num_vertices = 6
  /-- Assertion that the number of sides is equal to the number of vertices -/
  side_count : num_sides = num_vertices
  /-- Formula for the number of diagonals in a hexagon -/
  diagonal_count : num_diagonals = (num_vertices * (num_vertices - 3)) / 2

/-- The number of regions into which a regular hexagon is divided when all its sides and diagonals are drawn -/
def num_regions (h : RegularHexagonWithDiagonals) : Nat := 24

/-- Theorem stating that drawing all sides and diagonals of a regular hexagon divides it into 24 regions -/
theorem hexagon_division (h : RegularHexagonWithDiagonals) : num_regions h = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_division_l578_57816


namespace NUMINAMATH_CALUDE_probability_red_from_box2_is_11_27_l578_57867

/-- Represents a box containing balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from box 2 after the described process -/
def probability_red_from_box2 (box1 box2 : Box) : ℚ :=
  let total_balls1 := box1.white + box1.red
  let total_balls2 := box2.white + box2.red + 1
  let prob_white_from_box1 := box1.white / total_balls1
  let prob_red_from_box1 := box1.red / total_balls1
  let prob_red_if_white_moved := prob_white_from_box1 * (box2.red / total_balls2)
  let prob_red_if_red_moved := prob_red_from_box1 * ((box2.red + 1) / total_balls2)
  prob_red_if_white_moved + prob_red_if_red_moved

theorem probability_red_from_box2_is_11_27 :
  let box1 : Box := { white := 2, red := 4 }
  let box2 : Box := { white := 5, red := 3 }
  probability_red_from_box2 box1 box2 = 11/27 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_from_box2_is_11_27_l578_57867


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l578_57855

/-- The orthocenter of a triangle ABC in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (5/3, 29/3, 8/3) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, -1, 5)
  let C : ℝ × ℝ × ℝ := (1, 5, 2)
  orthocenter A B C = (5/3, 29/3, 8/3) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l578_57855


namespace NUMINAMATH_CALUDE_color_infinite_lines_parallelogram_property_coloring_theorem_l578_57836

-- Define the color type
inductive Color where
  | White : Color
  | Red : Color
  | Black : Color

-- Define the coloring function
def f : ℤ × ℤ → Color :=
  sorry

-- Condition 1: Each color appears on infinitely many horizontal lines
theorem color_infinite_lines :
  ∀ c : Color, ∃ (s : Set ℤ), Set.Infinite s ∧
    ∀ y ∈ s, ∃ (t : Set ℤ), Set.Infinite t ∧
      ∀ x ∈ t, f (x, y) = c :=
  sorry

-- Condition 2: Parallelogram property
theorem parallelogram_property :
  ∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Black →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ a + c = b + d :=
  sorry

-- Main theorem combining both conditions
theorem coloring_theorem :
  ∃ (f : ℤ × ℤ → Color),
    (∀ c : Color, ∃ (s : Set ℤ), Set.Infinite s ∧
      ∀ y ∈ s, ∃ (t : Set ℤ), Set.Infinite t ∧
        ∀ x ∈ t, f (x, y) = c) ∧
    (∀ a b c : ℤ × ℤ,
      f a = Color.White → f b = Color.Red → f c = Color.Black →
      ∃ d : ℤ × ℤ, f d = Color.Red ∧ a + c = b + d) :=
  sorry

end NUMINAMATH_CALUDE_color_infinite_lines_parallelogram_property_coloring_theorem_l578_57836


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l578_57878

/-- The ellipse defined by x^2/9 + y^2/4 = 1 -/
def ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 4) = 1}

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse defined by x^2/9 + y^2/4 = 1 is 6 -/
theorem ellipse_major_axis_length : 
  ∀ p ∈ ellipse, major_axis_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l578_57878
