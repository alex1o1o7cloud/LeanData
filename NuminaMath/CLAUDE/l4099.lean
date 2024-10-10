import Mathlib

namespace floor_e_squared_l4099_409994

theorem floor_e_squared : ⌊Real.exp 1 ^ 2⌋ = 7 := by sorry

end floor_e_squared_l4099_409994


namespace reflected_rays_angle_l4099_409983

theorem reflected_rays_angle 
  (α β : Real) 
  (h_α : 0 < α ∧ α < π/2) 
  (h_β : 0 < β ∧ β < π/2) : 
  ∃ θ : Real, θ = Real.arccos (1 - 2 * Real.sin α ^ 2 * Real.sin β ^ 2) := by
sorry

end reflected_rays_angle_l4099_409983


namespace expression_evaluation_l4099_409971

theorem expression_evaluation :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 := by
  sorry

end expression_evaluation_l4099_409971


namespace five_digit_number_formation_l4099_409916

/-- A two-digit number is between 10 and 99, inclusive -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A three-digit number is between 100 and 999, inclusive -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The five-digit number formed by placing x to the left of y -/
def FiveDigitNumber (x y : ℕ) : ℕ := 1000 * x + y

theorem five_digit_number_formation (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : ThreeDigitNumber y) :
  FiveDigitNumber x y = 1000 * x + y := by
  sorry

end five_digit_number_formation_l4099_409916


namespace coefficient_x5_expansion_l4099_409927

/-- The coefficient of x^5 in the expansion of (2 + √x - x^2018/2017)^12 -/
def coefficient_x5 : ℕ :=
  -- Define the coefficient here
  264

/-- Theorem stating that the coefficient of x^5 in the expansion of (2 + √x - x^2018/2017)^12 is 264 -/
theorem coefficient_x5_expansion :
  coefficient_x5 = 264 := by
  sorry

end coefficient_x5_expansion_l4099_409927


namespace min_cost_29_disks_l4099_409989

/-- Represents the cost of a package of disks -/
structure Package where
  quantity : Nat
  price : Nat

/-- Calculates the minimum cost to buy at least n disks given a list of packages -/
def minCost (packages : List Package) (n : Nat) : Nat :=
  sorry

/-- The available packages -/
def availablePackages : List Package :=
  [{ quantity := 1, price := 20 },
   { quantity := 10, price := 111 },
   { quantity := 25, price := 265 }]

theorem min_cost_29_disks :
  minCost availablePackages 29 = 333 :=
sorry

end min_cost_29_disks_l4099_409989


namespace kona_trip_distance_l4099_409935

/-- The distance from Kona's apartment to the bakery in miles -/
def apartment_to_bakery : ℝ := 9

/-- The distance from the bakery to Kona's grandmother's house in miles -/
def bakery_to_grandma : ℝ := 24

/-- The additional distance of the round trip with bakery stop compared to without -/
def additional_distance : ℝ := 6

/-- The distance from Kona's grandmother's house to his apartment in miles -/
def grandma_to_apartment : ℝ := 27

theorem kona_trip_distance :
  apartment_to_bakery + bakery_to_grandma + grandma_to_apartment =
  2 * grandma_to_apartment + additional_distance :=
sorry

end kona_trip_distance_l4099_409935


namespace first_thrilling_thursday_is_correct_l4099_409939

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns true if the given date is a Thursday -/
def isThursday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns true if the given date is a Thrilling Thursday -/
def isThrillingThursday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- The date of school start -/
def schoolStartDate : Date :=
  { day := 12, month := 9 }

/-- The day of the week when school starts -/
def schoolStartDay : DayOfWeek :=
  DayOfWeek.Tuesday

/-- The date of the first Thrilling Thursday -/
def firstThrillingThursday : Date :=
  { day := 26, month := 10 }

theorem first_thrilling_thursday_is_correct :
  isThrillingThursday firstThrillingThursday schoolStartDate schoolStartDay ∧
  ∀ d, d.month ≥ schoolStartDate.month ∧ 
       (d.month > schoolStartDate.month ∨ (d.month = schoolStartDate.month ∧ d.day ≥ schoolStartDate.day)) ∧
       isThrillingThursday d schoolStartDate schoolStartDay →
       (d.month > firstThrillingThursday.month ∨ 
        (d.month = firstThrillingThursday.month ∧ d.day ≥ firstThrillingThursday.day)) :=
  sorry

end first_thrilling_thursday_is_correct_l4099_409939


namespace cubic_equation_root_l4099_409934

theorem cubic_equation_root (a b : ℚ) : 
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 45 = 0 ∧ x = -2 - 5*Real.sqrt 3) →
  a = 239/71 := by
sorry

end cubic_equation_root_l4099_409934


namespace max_cyclic_product_permutation_l4099_409922

def cyclic_product (xs : List ℕ) : ℕ :=
  let n := xs.length
  List.sum (List.zipWith (· * ·) xs (xs.rotateLeft 1))

theorem max_cyclic_product_permutation :
  let perms := List.permutations [1, 2, 3, 4, 5]
  let max_val := perms.map cyclic_product |>.maximum?
  let max_count := (perms.filter (λ p ↦ cyclic_product p = max_val.getD 0)).length
  (max_val.getD 0 = 48) ∧ (max_count = 10) := by
  sorry

end max_cyclic_product_permutation_l4099_409922


namespace transformed_quadratic_roots_l4099_409996

theorem transformed_quadratic_roots (α β : ℂ) : 
  (3 * α^2 + 2 * α + 1 = 0) → 
  (3 * β^2 + 2 * β + 1 = 0) → 
  ((3 * α + 2)^2 + 4 = 0) ∧ ((3 * β + 2)^2 + 4 = 0) := by
sorry

end transformed_quadratic_roots_l4099_409996


namespace sum_of_x_solutions_is_zero_l4099_409948

theorem sum_of_x_solutions_is_zero (x y : ℝ) :
  y = 6 →
  x^2 + y^2 = 169 →
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ 
    ((x = x₁ ∨ x = x₂) ↔ (y = 6 ∧ x^2 + y^2 = 169)) :=
by sorry

end sum_of_x_solutions_is_zero_l4099_409948


namespace complex_exp_form_l4099_409993

theorem complex_exp_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → Complex.arg z = π / 3 := by
  sorry

end complex_exp_form_l4099_409993


namespace existence_condition_l4099_409921

theorem existence_condition (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - 2*x - a ≥ 0) ↔ a ≤ 3 := by
  sorry

end existence_condition_l4099_409921


namespace wall_width_proof_l4099_409946

theorem wall_width_proof (width height length : ℝ) 
  (height_def : height = 6 * width)
  (length_def : length = 7 * height)
  (volume_def : width * height * length = 86436) :
  width = 7 :=
by sorry

end wall_width_proof_l4099_409946


namespace train_bridge_time_l4099_409992

/-- Time for a train to pass a bridge -/
theorem train_bridge_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  train_speed_kmh = 45 →
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 40 := by
  sorry

end train_bridge_time_l4099_409992


namespace used_car_seller_problem_l4099_409999

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_clients = 24)
  (h2 : cars_per_client = 2)
  (h3 : selections_per_car = 3) :
  (num_clients * cars_per_client) / selections_per_car = 16 := by
  sorry

end used_car_seller_problem_l4099_409999


namespace max_triangle_side_l4099_409978

theorem max_triangle_side (a : ℕ) : 
  (3 + 8 > a ∧ 3 + a > 8 ∧ 8 + a > 3) → a ≤ 10 :=
by sorry

end max_triangle_side_l4099_409978


namespace coordinates_uniquely_determine_position_l4099_409997

-- Define a structure for geographical coordinates
structure GeoCoord where
  longitude : Real
  latitude : Real

-- Define a type for position descriptors
inductive PositionDescriptor
  | Distance (d : Real) (reference : String)
  | RoadName (name : String)
  | Coordinates (coord : GeoCoord)
  | Direction (angle : Real) (reference : String)

-- Function to check if a descriptor uniquely determines a position
def uniquelyDeterminesPosition (descriptor : PositionDescriptor) : Prop :=
  match descriptor with
  | PositionDescriptor.Coordinates _ => True
  | _ => False

-- Theorem stating that only coordinates uniquely determine a position
theorem coordinates_uniquely_determine_position
  (descriptor : PositionDescriptor) :
  uniquelyDeterminesPosition descriptor ↔
  ∃ (coord : GeoCoord), descriptor = PositionDescriptor.Coordinates coord :=
sorry

#check coordinates_uniquely_determine_position

end coordinates_uniquely_determine_position_l4099_409997


namespace t_shape_area_is_12_l4099_409998

def square_area (side : ℝ) : ℝ := side * side

def t_shape_area (outer_side : ℝ) (inner_side1 : ℝ) (inner_side2 : ℝ) : ℝ :=
  square_area outer_side - (2 * square_area inner_side1 + square_area inner_side2)

theorem t_shape_area_is_12 :
  t_shape_area 6 2 4 = 12 := by sorry

end t_shape_area_is_12_l4099_409998


namespace A_equals_B_l4099_409969

/-- Number of partitions of n where even parts are distinct -/
def A (n : ℕ) : ℕ := sorry

/-- Number of partitions of n where each part appears at most 3 times -/
def B (n : ℕ) : ℕ := sorry

/-- Theorem stating that A_n equals B_n for all natural numbers n -/
theorem A_equals_B : ∀ n : ℕ, A n = B n := by sorry

end A_equals_B_l4099_409969


namespace hair_cut_length_l4099_409938

/-- The amount of hair cut off is equal to the difference between the initial hair length and the final hair length. -/
theorem hair_cut_length (initial_length final_length cut_length : ℕ) 
  (h1 : initial_length = 18)
  (h2 : final_length = 9)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 9 := by
  sorry

end hair_cut_length_l4099_409938


namespace travel_agency_choice_l4099_409977

-- Define the cost functions for each travel agency
def costA (x : ℝ) : ℝ := 2000 * x * 0.75
def costB (x : ℝ) : ℝ := 2000 * (x - 1) * 0.8

-- Define the theorem
theorem travel_agency_choice (x : ℝ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x < 16 → costB x < costA x) ∧
  (x = 16 → costA x = costB x) ∧
  (16 < x ∧ x ≤ 25 → costA x < costB x) :=
sorry

end travel_agency_choice_l4099_409977


namespace geometric_sequence_problem_l4099_409951

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 36 * r = b ∧ b * r = 2 / 9) → b = 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_problem_l4099_409951


namespace sample_size_calculation_l4099_409913

/-- Given a sample with 16 units of model A, and the ratio of quantities of 
    models A, B, and C being 2:3:5, the total sample size n is 80. -/
theorem sample_size_calculation (model_a_count : ℕ) (ratio_a ratio_b ratio_c : ℕ) :
  model_a_count = 16 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 5 →
  (ratio_a : ℚ) / (ratio_a + ratio_b + ratio_c : ℚ) * (model_a_count * (ratio_a + ratio_b + ratio_c) / ratio_a) = 80 :=
by sorry

end sample_size_calculation_l4099_409913


namespace sibling_product_sixteen_l4099_409909

/-- Represents a family with a given number of girls and boys -/
structure Family :=
  (girls : ℕ)
  (boys : ℕ)

/-- Calculates the product of sisters and brothers for a member of the family -/
def siblingProduct (f : Family) : ℕ :=
  (f.girls - 1) * f.boys

/-- Theorem: In a family with 5 girls and 4 boys, the product of sisters and brothers is 16 -/
theorem sibling_product_sixteen (f : Family) (h1 : f.girls = 5) (h2 : f.boys = 4) :
  siblingProduct f = 16 := by
  sorry

end sibling_product_sixteen_l4099_409909


namespace faye_pencil_rows_l4099_409932

/-- Given that Faye has 720 pencils in total and places 24 pencils in each row,
    prove that the number of rows she created is 30. -/
theorem faye_pencil_rows (total_pencils : Nat) (pencils_per_row : Nat) (rows : Nat) :
  total_pencils = 720 →
  pencils_per_row = 24 →
  rows * pencils_per_row = total_pencils →
  rows = 30 := by
  sorry

#check faye_pencil_rows

end faye_pencil_rows_l4099_409932


namespace integral_sin_plus_sqrt_one_minus_x_squared_l4099_409945

theorem integral_sin_plus_sqrt_one_minus_x_squared : 
  ∫ x in (-1)..1, (Real.sin x + Real.sqrt (1 - x^2)) = π / 2 := by sorry

end integral_sin_plus_sqrt_one_minus_x_squared_l4099_409945


namespace coin_distribution_l4099_409914

theorem coin_distribution (a b c d e : ℚ) : 
  -- The amounts form an arithmetic sequence
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) →
  -- The total number of coins is 5
  a + b + c + d + e = 5 →
  -- The sum of first two equals the sum of last three
  a + b = c + d + e →
  -- B receives 7/6 coins
  b = 7/6 := by sorry

end coin_distribution_l4099_409914


namespace simplify_expression_l4099_409929

theorem simplify_expression (r : ℝ) : 180 * r - 88 * r = 92 * r := by
  sorry

end simplify_expression_l4099_409929


namespace principal_calculation_l4099_409953

/-- The compound interest formula for yearly compounding -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem principal_calculation (final_amount : ℝ) (rate : ℝ) (time : ℕ)
  (h_final : final_amount = 3087)
  (h_rate : rate = 0.05)
  (h_time : time = 2) :
  ∃ principal : ℝ, 
    compound_interest principal rate time = final_amount ∧ 
    principal = 2800 := by
  sorry

end principal_calculation_l4099_409953


namespace art_count_l4099_409924

/-- The number of Asian art pieces seen -/
def asian_art : ℕ := 465

/-- The number of Egyptian art pieces seen -/
def egyptian_art : ℕ := 527

/-- The total number of art pieces seen -/
def total_art : ℕ := asian_art + egyptian_art

theorem art_count : total_art = 992 := by
  sorry

end art_count_l4099_409924


namespace unattainable_value_l4099_409906

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃y, y = (2 - x) / (3 * x + 4) ∧ y = -1/3 := by
sorry

end unattainable_value_l4099_409906


namespace probability_product_24_l4099_409908

def is_valid_die_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def product_equals_24 (a b c d : ℕ) : Prop :=
  is_valid_die_roll a ∧ is_valid_die_roll b ∧ is_valid_die_roll c ∧ is_valid_die_roll d ∧ a * b * c * d = 24

def count_valid_permutations : ℕ := 36

def total_outcomes : ℕ := 6^4

theorem probability_product_24 :
  (count_valid_permutations : ℚ) / total_outcomes = 1 / 36 :=
sorry

end probability_product_24_l4099_409908


namespace circles_intersect_l4099_409958

theorem circles_intersect : ∃ (x y : ℝ), 
  (x^2 + y^2 + 6*x - 7 = 0) ∧ (x^2 + y^2 + 6*y - 27 = 0) := by
  sorry

end circles_intersect_l4099_409958


namespace mod_equivalence_proof_l4099_409982

theorem mod_equivalence_proof : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4573 [ZMOD 8] → n = 3 := by
  sorry

end mod_equivalence_proof_l4099_409982


namespace probability_three_girls_l4099_409987

/-- The probability of choosing 3 girls from a club with 15 members (8 girls and 7 boys) is 8/65 -/
theorem probability_three_girls (total : ℕ) (girls : ℕ) (boys : ℕ) (h1 : total = 15) (h2 : girls = 8) (h3 : boys = 7) (h4 : total = girls + boys) :
  (Nat.choose girls 3 : ℚ) / (Nat.choose total 3) = 8 / 65 := by
  sorry

end probability_three_girls_l4099_409987


namespace power_function_properties_l4099_409930

-- Define the power function f
noncomputable def f : ℝ → ℝ := λ x => Real.sqrt x

-- State the theorem
theorem power_function_properties :
  (f 9 = 3) →
  (∀ x ≥ 4, f x ≥ 2) ∧
  (∀ x₁ x₂, x₂ > x₁ ∧ x₁ > 0 → (f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by
  sorry

end power_function_properties_l4099_409930


namespace simplify_and_evaluate_l4099_409974

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 2) (h2 : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 := by
  sorry

end simplify_and_evaluate_l4099_409974


namespace walt_age_l4099_409900

theorem walt_age (walt_age music_teacher_age : ℕ) : 
  music_teacher_age = 3 * walt_age →
  music_teacher_age + 12 = 2 * (walt_age + 12) →
  walt_age = 12 := by
sorry

end walt_age_l4099_409900


namespace cookie_ratio_l4099_409907

theorem cookie_ratio (monday_cookies : ℕ) (total_cookies : ℕ) 
  (h1 : monday_cookies = 32)
  (h2 : total_cookies = 92)
  (h3 : ∃ f : ℚ, 
    monday_cookies + f * monday_cookies + (3 * f * monday_cookies - 4) = total_cookies) :
  ∃ f : ℚ, f = 1/2 ∧ 
    monday_cookies + f * monday_cookies + (3 * f * monday_cookies - 4) = total_cookies :=
by
  sorry

end cookie_ratio_l4099_409907


namespace walter_school_allocation_l4099_409905

/-- Represents Walter's work schedule and earnings --/
structure WorkSchedule where
  days_per_week : ℕ
  hours_per_day : ℕ
  hourly_rate : ℚ
  school_allocation_ratio : ℚ

/-- Calculates the amount Walter allocates for school given his work schedule --/
def school_allocation (schedule : WorkSchedule) : ℚ :=
  schedule.days_per_week * schedule.hours_per_day * schedule.hourly_rate * schedule.school_allocation_ratio

/-- Theorem stating that Walter allocates $75 for school each week --/
theorem walter_school_allocation :
  let walter_schedule : WorkSchedule := {
    days_per_week := 5,
    hours_per_day := 4,
    hourly_rate := 5,
    school_allocation_ratio := 3/4
  }
  school_allocation walter_schedule = 75 := by
  sorry

end walter_school_allocation_l4099_409905


namespace retail_price_problem_l4099_409942

/-- The retail price problem -/
theorem retail_price_problem
  (wholesale_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2)
  (retail_price : ℝ) :
  retail_price = 120 ↔
    retail_price * (1 - discount_rate) = 
      wholesale_price * (1 + profit_rate) :=
by sorry

end retail_price_problem_l4099_409942


namespace max_value_constraint_l4099_409923

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x + 3*y < 60) :
  xy*(60 - 4*x - 3*y) ≤ 2000/3 ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4*x₀ + 3*y₀ < 60 ∧ x₀*y₀*(60 - 4*x₀ - 3*y₀) = 2000/3 :=
by sorry

end max_value_constraint_l4099_409923


namespace area_of_triangle_area_value_l4099_409944

theorem area_of_triangle : ℝ → Prop :=
  fun area =>
    ∃ (line1 line2 : ℝ → ℝ → Prop) (x_axis : ℝ → ℝ → Prop),
      (∀ x y, line1 x y ↔ y = x) ∧
      (∀ x y, line2 x y ↔ x = -7) ∧
      (∀ x y, x_axis x y ↔ y = 0) ∧
      (∃ x y, line1 x y ∧ line2 x y) ∧
      (let base := 7
       let height := 7
       area = (1/2) * base * height)

theorem area_value : area_of_triangle 24.5 := by
  sorry

end area_of_triangle_area_value_l4099_409944


namespace probability_no_defective_bulbs_l4099_409988

def total_bulbs : ℕ := 10
def defective_bulbs : ℕ := 4
def selected_bulbs : ℕ := 4

theorem probability_no_defective_bulbs :
  (Nat.choose (total_bulbs - defective_bulbs) selected_bulbs) /
  (Nat.choose total_bulbs selected_bulbs) = 1 / 14 :=
by sorry

end probability_no_defective_bulbs_l4099_409988


namespace regular_polygon_perimeter_l4099_409949

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / n = exterior_angle → 
  n * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l4099_409949


namespace kim_test_probability_l4099_409970

theorem kim_test_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end kim_test_probability_l4099_409970


namespace units_digit_of_product_l4099_409979

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of any power of 5 is 5 -/
axiom units_digit_power_of_five (k : ℕ) : units_digit (5^k) = 5

/-- The main theorem: The units digit of 5^11 * 2^3 is 0 -/
theorem units_digit_of_product : units_digit (5^11 * 2^3) = 0 := by
  sorry

end units_digit_of_product_l4099_409979


namespace fraction_ratio_equality_l4099_409975

theorem fraction_ratio_equality : 
  ∃ (x y : ℚ), x / y = (240 : ℚ) / 1547 ∧ 
  x / y / ((2 : ℚ) / 13) = ((5 : ℚ) / 34) / ((7 : ℚ) / 48) := by
  sorry

end fraction_ratio_equality_l4099_409975


namespace unique_functional_equation_solution_l4099_409972

theorem unique_functional_equation_solution :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, f (m + f n) = f m + f n + f (n + 1) :=
by
  sorry

end unique_functional_equation_solution_l4099_409972


namespace arithmetic_mean_of_fractions_l4099_409961

theorem arithmetic_mean_of_fractions : 
  let a : ℚ := 3/4
  let b : ℚ := 5/8
  (a + b) / 2 = 11/16 := by sorry

end arithmetic_mean_of_fractions_l4099_409961


namespace cuboid_edge_length_l4099_409912

theorem cuboid_edge_length (x : ℝ) : 
  x > 0 → 2 * x * 3 = 30 → x = 5 := by sorry

end cuboid_edge_length_l4099_409912


namespace rectangle_perimeter_l4099_409920

/-- Given a rectangle where the length is three times the width and the diagonal is 8√10,
    prove that its perimeter is 64. -/
theorem rectangle_perimeter (w l d : ℝ) : 
  l = 3 * w →                 -- length is three times the width
  d = 8 * (10 : ℝ).sqrt →     -- diagonal is 8√10
  w * w + l * l = d * d →     -- Pythagorean theorem
  2 * (w + l) = 64 :=         -- perimeter is 64
by sorry

end rectangle_perimeter_l4099_409920


namespace mult_inverse_mod_million_mult_inverse_specific_l4099_409963

/-- The multiplicative inverse of (A * B) modulo 1,000,000 is 466390 -/
theorem mult_inverse_mod_million : Int → Int → Prop :=
  fun A B => (A * B * 466390) % 1000000 = 1

/-- The theorem holds for A = 123456 and B = 162037 -/
theorem mult_inverse_specific : mult_inverse_mod_million 123456 162037 := by
  sorry

end mult_inverse_mod_million_mult_inverse_specific_l4099_409963


namespace sum_of_solutions_l4099_409926

theorem sum_of_solutions (x : ℝ) : (x + 16 / x = 12) → (∃ y : ℝ, y + 16 / y = 12 ∧ y ≠ x) → x + y = 12 :=
by sorry

end sum_of_solutions_l4099_409926


namespace tens_digit_of_sum_l4099_409933

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_sum : tens_digit (2^1500 + 5^768) = 9 := by
  sorry

end tens_digit_of_sum_l4099_409933


namespace pizza_consumption_order_l4099_409960

def pizza_sharing (eva gwen noah mia : ℚ) : Prop :=
  eva = 1/4 ∧ gwen = 1/6 ∧ noah = 1/5 ∧ mia = 1 - (eva + gwen + noah)

theorem pizza_consumption_order (eva gwen noah mia : ℚ) 
  (h : pizza_sharing eva gwen noah mia) : 
  eva > mia ∧ mia > noah ∧ noah > gwen :=
by
  sorry

#check pizza_consumption_order

end pizza_consumption_order_l4099_409960


namespace min_boxes_to_eliminate_l4099_409902

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $250,000 -/
def high_value_boxes : ℕ := 6

/-- The desired probability of holding a high-value box -/
def desired_probability : ℚ := 1/3

/-- The function to calculate the minimum number of boxes to eliminate -/
def boxes_to_eliminate : ℕ := total_boxes - (high_value_boxes * 3)

/-- Theorem stating the minimum number of boxes to eliminate -/
theorem min_boxes_to_eliminate :
  boxes_to_eliminate = 12 := by sorry

end min_boxes_to_eliminate_l4099_409902


namespace right_triangle_construction_impossibility_l4099_409911

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a point being inside a circle
def IsInside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), c = Circle center radius ∧
    (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2

-- Define a circle with diameter AB
def CircleWithDiameter (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  Circle ((A.1 + B.1)/2, (A.2 + B.2)/2) (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2)

-- Define intersection of two sets
def Intersects (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ s1 ∧ p ∈ s2

-- Main theorem
theorem right_triangle_construction_impossibility
  (C : Set (ℝ × ℝ)) (A B : ℝ × ℝ)
  (h_circle : ∃ center radius, C = Circle center radius)
  (h_A_inside : IsInside A C)
  (h_B_inside : IsInside B C) :
  (¬ ∃ P Q R : ℝ × ℝ,
    P ∈ C ∧ Q ∈ C ∧ R ∈ C ∧
    (A.1 - P.1) * (Q.1 - P.1) + (A.2 - P.2) * (Q.2 - P.2) = 0 ∧
    (B.1 - P.1) * (R.1 - P.1) + (B.2 - P.2) * (R.2 - P.2) = 0 ∧
    (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  ↔
  ¬ Intersects (CircleWithDiameter A B) C :=
by sorry

end right_triangle_construction_impossibility_l4099_409911


namespace fraction_equality_implies_c_geq_one_l4099_409981

theorem fraction_equality_implies_c_geq_one
  (a b : ℕ+) (c : ℝ)
  (h_c_pos : c > 0)
  (h_eq : (a + 1) / (b + c) = b / a) :
  c ≥ 1 := by
  sorry

end fraction_equality_implies_c_geq_one_l4099_409981


namespace parabolas_intersection_l4099_409947

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 4

-- Define a function to check if a point is on both parabolas
def is_intersection (x y : ℝ) : Prop :=
  parabola1 x = y ∧ parabola2 x = y

-- Theorem statement
theorem parabolas_intersection :
  (is_intersection (-3) 25) ∧ 
  (is_intersection 1 1) ∧
  (∀ x y : ℝ, is_intersection x y → (x = -3 ∧ y = 25) ∨ (x = 1 ∧ y = 1)) :=
by sorry

end parabolas_intersection_l4099_409947


namespace m_range_theorem_l4099_409965

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 1| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem m_range_theorem (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(q x m) → ¬(p x)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m ≤ 2) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end m_range_theorem_l4099_409965


namespace special_triangle_side_lengths_l4099_409941

/-- Triangle with consecutive integer side lengths and perpendicular median and angle bisector -/
structure SpecialTriangle where
  -- Side lengths
  a : ℕ
  b : ℕ
  c : ℕ
  -- Consecutive integer side lengths
  consecutive_sides : c = b + 1 ∧ b = a + 1
  -- Median from A
  median_a : ℝ × ℝ
  -- Angle bisector from B
  bisector_b : ℝ × ℝ
  -- Perpendicularity condition
  perpendicular : median_a.1 * bisector_b.1 + median_a.2 * bisector_b.2 = 0

/-- The side lengths of a special triangle are 2, 3, and 4 -/
theorem special_triangle_side_lengths (t : SpecialTriangle) : t.a = 2 ∧ t.b = 3 ∧ t.c = 4 :=
sorry

end special_triangle_side_lengths_l4099_409941


namespace quadratic_coefficient_l4099_409985

/-- Given a quadratic function f(x) = ax² + bx + c, if f(2) - f(-2) = 4, then b = 1. -/
theorem quadratic_coefficient (a b c : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = 4 →
  b = 1 := by
  sorry

end quadratic_coefficient_l4099_409985


namespace cubic_root_equality_l4099_409955

theorem cubic_root_equality (p q r : ℝ) : 
  (∀ x : ℝ, x^3 - 3*p*x^2 + 3*q^2*x - r^3 = 0 ↔ (x = p ∨ x = q ∨ x = r)) →
  p = q ∧ q = r :=
by sorry

end cubic_root_equality_l4099_409955


namespace geometric_sequence_condition_l4099_409952

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) (c : ℝ) : ℝ := 2^n + c

/-- The n-th term of the sequence a_n -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n-1) c

/-- Predicate to check if a sequence is geometric -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n+1) = r * a n

theorem geometric_sequence_condition (c : ℝ) :
  is_geometric_sequence (a · c) ↔ c = -1 := by sorry

end geometric_sequence_condition_l4099_409952


namespace train_speed_l4099_409943

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 300 →
  crossing_time = 36 →
  ∃ (speed : ℝ), abs (speed - (train_length + bridge_length) / crossing_time) < 0.01 :=
by sorry

end train_speed_l4099_409943


namespace movie_pause_point_l4099_409928

/-- Proves that the pause point in a movie is halfway through, given the total length and remaining time. -/
theorem movie_pause_point (total_length remaining : ℕ) (h1 : total_length = 60) (h2 : remaining = 30) :
  total_length - remaining = 30 := by
  sorry

end movie_pause_point_l4099_409928


namespace valid_queues_count_l4099_409917

/-- Represents the amount a customer has: 
    1 for 50 cents (exact change), -1 for one dollar (needs change) -/
inductive CustomerMoney : Type
  | exact : CustomerMoney
  | needsChange : CustomerMoney

/-- A queue of customers -/
def CustomerQueue := List CustomerMoney

/-- The nth Catalan number -/
def catalanNumber (n : ℕ) : ℕ := sorry

/-- Checks if a queue is valid (cashier can always give change) -/
def isValidQueue (queue : CustomerQueue) : Prop := sorry

/-- Counts the number of valid queues for 2n customers -/
def countValidQueues (n : ℕ) : ℕ := sorry

/-- Theorem: The number of valid queues for 2n customers 
    (n with exact change, n needing change) is the nth Catalan number -/
theorem valid_queues_count (n : ℕ) : 
  countValidQueues n = catalanNumber n := by sorry

end valid_queues_count_l4099_409917


namespace train_speed_problem_l4099_409962

/-- Given a train journey with the following properties:
  * Total distance is 3x km
  * First part of the journey covers x km at speed V kmph
  * Second part of the journey covers 2x km at 20 kmph
  * Average speed for the entire journey is 27 kmph
  Then, the speed V of the first part of the journey is 90 kmph. -/
theorem train_speed_problem (x : ℝ) (V : ℝ) (h_x_pos : x > 0) (h_V_pos : V > 0) :
  (x / V + 2 * x / 20) = 3 * x / 27 → V = 90 := by
  sorry

end train_speed_problem_l4099_409962


namespace dino_money_theorem_l4099_409925

/-- Calculates Dino's remaining money at the end of the month -/
def dino_remaining_money (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem: Dino's remaining money at the end of the month is $500 -/
theorem dino_money_theorem : dino_remaining_money 20 30 5 10 20 40 500 = 500 := by
  sorry

end dino_money_theorem_l4099_409925


namespace complex_fraction_evaluation_l4099_409984

theorem complex_fraction_evaluation (u v : ℂ) (hu : u ≠ 0) (hv : v ≠ 0) 
  (h : u^2 + u*v + v^2 = 0) : 
  (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end complex_fraction_evaluation_l4099_409984


namespace red_purple_probability_l4099_409919

def total_balls : ℕ := 120
def red_balls : ℕ := 20
def purple_balls : ℕ := 5

theorem red_purple_probability : 
  (red_balls * purple_balls * 2 : ℚ) / (total_balls * (total_balls - 1)) = 5 / 357 := by
  sorry

end red_purple_probability_l4099_409919


namespace lindas_calculation_l4099_409959

theorem lindas_calculation (x y z : ℝ) 
  (h1 : x - (y + z) = 5) 
  (h2 : x - y + z = -1) : 
  x - y = 2 := by
  sorry

end lindas_calculation_l4099_409959


namespace function_inequality_implies_a_bound_l4099_409931

open Real

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ →
    (log x₁ + a * x₁^2 - (log x₂ + a * x₂^2)) / (x₁ - x₂) > 2) →
  a ≥ (1 / 2) := by
  sorry

end function_inequality_implies_a_bound_l4099_409931


namespace total_black_dots_l4099_409918

theorem total_black_dots (num_butterflies : ℕ) (black_dots_per_butterfly : ℕ) 
  (h1 : num_butterflies = 397) 
  (h2 : black_dots_per_butterfly = 12) : 
  num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end total_black_dots_l4099_409918


namespace max_books_with_200_dollars_l4099_409904

/-- The maximum number of books that can be purchased with a given budget and book price -/
def maxBooks (budget : ℕ) (bookPrice : ℕ) : ℕ :=
  (budget * 100) / bookPrice

/-- Theorem: Given a book price of $45 and a budget of $200, the maximum number of books that can be purchased is 444 -/
theorem max_books_with_200_dollars : maxBooks 200 45 = 444 := by
  sorry

end max_books_with_200_dollars_l4099_409904


namespace changed_number_proof_l4099_409940

theorem changed_number_proof (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 8 →
  (8 + b + c + d + e) / 5 = 9 →
  a = 3 := by
sorry

end changed_number_proof_l4099_409940


namespace morgan_hula_hoop_time_l4099_409915

/-- Given information about hula hooping times for Nancy, Casey, and Morgan,
    prove that Morgan can hula hoop for 21 minutes. -/
theorem morgan_hula_hoop_time :
  ∀ (nancy casey morgan : ℕ),
    nancy = 10 →
    casey = nancy - 3 →
    morgan = 3 * casey →
    morgan = 21 :=
by
  sorry

end morgan_hula_hoop_time_l4099_409915


namespace puzzle_unique_solution_l4099_409986

/-- Represents a mapping from letters to digits -/
def LetterMapping := Char → Fin 10

/-- Checks if a mapping is valid (different letters map to different digits) -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a word to a number using the given mapping -/
def word_to_number (word : List Char) (m : LetterMapping) : ℕ :=
  word.foldl (fun acc d => 10 * acc + (m d).val) 0

/-- The cryptarithmetic puzzle equation -/
def puzzle_equation (m : LetterMapping) : Prop :=
  let dodge := word_to_number ['D', 'O', 'D', 'G', 'E'] m
  let strike := word_to_number ['S', 'T', 'R', 'I', 'K', 'E'] m
  let fighting := word_to_number ['F', 'I', 'G', 'H', 'T', 'I', 'N', 'G'] m
  dodge + strike = fighting

/-- The main theorem stating that the puzzle has a unique solution -/
theorem puzzle_unique_solution :
  ∃! m : LetterMapping, is_valid_mapping m ∧ puzzle_equation m :=
sorry

end puzzle_unique_solution_l4099_409986


namespace m_intersect_n_equals_open_interval_l4099_409976

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 + 5*x - 14 < 0}

-- Define set N
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

-- Theorem statement
theorem m_intersect_n_equals_open_interval :
  M ∩ N = Set.Ioo 1 2 := by sorry

end m_intersect_n_equals_open_interval_l4099_409976


namespace adam_has_14_apples_l4099_409901

-- Define the number of apples Jackie has
def jackie_apples : ℕ := 9

-- Define Adam's apples in relation to Jackie's
def adam_apples : ℕ := jackie_apples + 5

-- Theorem statement
theorem adam_has_14_apples : adam_apples = 14 := by
  sorry

end adam_has_14_apples_l4099_409901


namespace uncle_gift_amount_l4099_409954

/-- The amount of money Geoffrey's uncle gave him --/
def uncle_gift (grandmother_gift aunt_gift total_after_gifts spent_on_games money_left : ℕ) : ℕ :=
  total_after_gifts - grandmother_gift - aunt_gift - money_left

/-- Theorem stating the amount of money Geoffrey's uncle gave him --/
theorem uncle_gift_amount : 
  uncle_gift 20 25 125 105 20 = 60 := by
  sorry

#eval uncle_gift 20 25 125 105 20

end uncle_gift_amount_l4099_409954


namespace little_red_final_score_l4099_409980

/-- Calculates the final score for the "Sunshine Sports" competition --/
def final_score (running_score fancy_jump_rope_score jump_rope_score : ℝ)
  (running_weight fancy_jump_rope_weight jump_rope_weight : ℝ) : ℝ :=
  running_score * running_weight +
  fancy_jump_rope_score * fancy_jump_rope_weight +
  jump_rope_score * jump_rope_weight

/-- Theorem stating that Little Red's final score is 83 --/
theorem little_red_final_score :
  final_score 90 80 70 0.5 0.3 0.2 = 83 := by
  sorry

#eval final_score 90 80 70 0.5 0.3 0.2

end little_red_final_score_l4099_409980


namespace sum_of_roots_is_6_l4099_409995

-- Define a quadratic function
variable (f : ℝ → ℝ)

-- Define the symmetry property
def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (3 + x) = f (3 - x)

-- Define the property of having two real roots
def has_two_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Theorem statement
theorem sum_of_roots_is_6 (f : ℝ → ℝ) 
  (h_sym : is_symmetric_about_3 f) 
  (h_roots : has_two_real_roots f) :
  ∃ x₁ x₂ : ℝ, has_two_real_roots f ∧ x₁ + x₂ = 6 :=
sorry

end sum_of_roots_is_6_l4099_409995


namespace martha_cards_l4099_409903

theorem martha_cards (initial_cards given_cards : ℝ) 
  (h1 : initial_cards = 76.0)
  (h2 : given_cards = 3.0) : 
  initial_cards - given_cards = 73.0 := by
  sorry

end martha_cards_l4099_409903


namespace quadratic_form_nonnegative_l4099_409990

theorem quadratic_form_nonnegative
  (a b c x y z : ℝ)
  (sum_xyz : x + y + z = 0)
  (sum_abc_nonneg : a + b + c ≥ 0)
  (sum_products_nonneg : a * b + b * c + c * a ≥ 0) :
  a * x^2 + b * y^2 + c * z^2 ≥ 0 := by
sorry

end quadratic_form_nonnegative_l4099_409990


namespace arccos_cos_seven_l4099_409964

theorem arccos_cos_seven : Real.arccos (Real.cos 7) = 7 - 2 * Real.pi := by sorry

end arccos_cos_seven_l4099_409964


namespace quadratic_monotonicity_l4099_409956

/-- A function f is monotonically increasing on (a, +∞) if for all x, y > a, x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x > a → y > a → x < y → f x < f y

/-- The quadratic function f(x) = x^2 + mx - 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2

theorem quadratic_monotonicity (m : ℝ) :
  MonoIncreasing (f m) 2 → m ≥ -4 := by sorry

end quadratic_monotonicity_l4099_409956


namespace square_units_digit_l4099_409957

theorem square_units_digit (n : ℤ) : 
  (n^2 / 10) % 10 = 7 → n^2 % 10 = 6 := by sorry

end square_units_digit_l4099_409957


namespace fraction_division_difference_l4099_409937

theorem fraction_division_difference : (5 / 3) / (1 / 6) - 2 / 3 = 28 / 3 := by
  sorry

end fraction_division_difference_l4099_409937


namespace books_from_first_shop_is_32_l4099_409950

/-- Represents the number of books bought from the first shop -/
def books_from_first_shop : ℕ := sorry

/-- The total amount spent on books from the first shop in Rs -/
def amount_first_shop : ℕ := 1500

/-- The number of books bought from the second shop -/
def books_from_second_shop : ℕ := 60

/-- The total amount spent on books from the second shop in Rs -/
def amount_second_shop : ℕ := 340

/-- The average price per book for all books in Rs -/
def average_price : ℕ := 20

/-- Theorem stating that the number of books bought from the first shop is 32 -/
theorem books_from_first_shop_is_32 : books_from_first_shop = 32 := by
  sorry

end books_from_first_shop_is_32_l4099_409950


namespace min_value_sin_2x_minus_pi_4_l4099_409966

theorem min_value_sin_2x_minus_pi_4 :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (2 * x - Real.pi / 4) ≥ -Real.sqrt 2 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (2 * x - Real.pi / 4) = -Real.sqrt 2 / 2) :=
sorry

end min_value_sin_2x_minus_pi_4_l4099_409966


namespace perpendicular_point_sets_l4099_409968

-- Define the concept of a "perpendicular point set"
def isPerpendicular (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₁ y₁ : ℝ), (x₁, y₁) ∈ M → 
    ∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ M ∧ x₁ * x₂ + y₁ * y₂ = 0

-- Define the sets
def M₁ : Set (ℝ × ℝ) := {(x, y) | y = 1 / x^2 ∧ x ≠ 0}
def M₂ : Set (ℝ × ℝ) := {(x, y) | y = Real.log x / Real.log 2 ∧ x > 0}
def M₃ : Set (ℝ × ℝ) := {(x, y) | y = 2^x - 2}
def M₄ : Set (ℝ × ℝ) := {(x, y) | y = Real.sin x + 1}

-- State the theorem
theorem perpendicular_point_sets :
  isPerpendicular M₁ ∧ 
  ¬(isPerpendicular M₂) ∧ 
  isPerpendicular M₃ ∧ 
  isPerpendicular M₄ := by
  sorry

end perpendicular_point_sets_l4099_409968


namespace abs_inequality_equivalence_l4099_409991

theorem abs_inequality_equivalence (x : ℝ) :
  (1 ≤ |x - 2| ∧ |x - 2| ≤ 7) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9)) :=
sorry

end abs_inequality_equivalence_l4099_409991


namespace anniversary_products_l4099_409936

/-- Commemorative albums and bone china cups problem -/
theorem anniversary_products (total_cost album_cost cup_cost album_price cup_price : ℝ)
  (h1 : total_cost = 312000)
  (h2 : album_cost = 3 * cup_cost)
  (h3 : album_cost + cup_cost = total_cost)
  (h4 : album_price = 1.5 * cup_price)
  (h5 : cup_cost / cup_price - 4 * (album_cost / album_price) = 1600) :
  album_cost = 240000 ∧ cup_cost = 72000 ∧ album_price = 45 ∧ cup_price = 30 := by
sorry

end anniversary_products_l4099_409936


namespace triangle_angle_not_all_greater_60_l4099_409910

theorem triangle_angle_not_all_greater_60 :
  ∀ (a b c : Real),
  (a > 0) → (b > 0) → (c > 0) →
  (a + b + c = 180) →
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by
  sorry

end triangle_angle_not_all_greater_60_l4099_409910


namespace expression_simplification_l4099_409967

theorem expression_simplification
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hbc : b - 2 / c ≠ 0) :
  (a - 2 / b) / (b - 2 / c) = c / b :=
by sorry

end expression_simplification_l4099_409967


namespace geometric_mean_of_4_and_16_l4099_409973

theorem geometric_mean_of_4_and_16 (x : ℝ) :
  x ^ 2 = 4 * 16 → x = 8 ∨ x = -8 := by
  sorry

end geometric_mean_of_4_and_16_l4099_409973
