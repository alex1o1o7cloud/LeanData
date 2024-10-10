import Mathlib

namespace boxwoods_shaped_into_spheres_l1540_154021

/-- Calculates the number of boxwoods shaped into spheres given the total number of boxwoods,
    costs for trimming and shaping, and the total charge. -/
theorem boxwoods_shaped_into_spheres
  (total_boxwoods : ℕ)
  (trim_cost : ℚ)
  (shape_cost : ℚ)
  (total_charge : ℚ)
  (h1 : total_boxwoods = 30)
  (h2 : trim_cost = 5)
  (h3 : shape_cost = 15)
  (h4 : total_charge = 210) :
  (total_charge - (total_boxwoods * trim_cost)) / shape_cost = 4 :=
by sorry

end boxwoods_shaped_into_spheres_l1540_154021


namespace janes_bagels_l1540_154015

theorem janes_bagels (muffin_cost bagel_cost : ℕ) (total_days : ℕ) : 
  muffin_cost = 60 →
  bagel_cost = 80 →
  total_days = 7 →
  ∃! (num_bagels : ℕ), 
    num_bagels ≤ total_days ∧
    ∃ (total_cost : ℕ), 
      total_cost * 100 = num_bagels * bagel_cost + (total_days - num_bagels) * muffin_cost ∧
      num_bagels = 4 :=
by sorry

end janes_bagels_l1540_154015


namespace triangle_inequality_with_circumradius_and_altitudes_l1540_154018

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  -- Add conditions to ensure it's a valid triangle
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  pos_R : 0 < R
  pos_altitudes : 0 < h_a ∧ 0 < h_b ∧ 0 < h_c

-- State the theorem
theorem triangle_inequality_with_circumradius_and_altitudes (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 2 * t.R * (t.h_a + t.h_b + t.h_c) := by
  sorry

end triangle_inequality_with_circumradius_and_altitudes_l1540_154018


namespace polynomial_divisibility_l1540_154049

theorem polynomial_divisibility (a b c d e : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 7 * k) →
  (∃ k₁ k₂ k₃ k₄ k₅ : ℤ, a = 7 * k₁ ∧ b = 7 * k₂ ∧ c = 7 * k₃ ∧ d = 7 * k₄ ∧ e = 7 * k₅) :=
by sorry

end polynomial_divisibility_l1540_154049


namespace rent_increase_percentage_l1540_154077

/-- Given Elaine's rent spending patterns over two years, prove that this year's rent
    is 187.5% of last year's rent. -/
theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.25 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 187.5 := by
  sorry

end rent_increase_percentage_l1540_154077


namespace bicycle_cost_l1540_154088

def hourly_rate : ℕ := 5
def monday_hours : ℕ := 2
def wednesday_hours : ℕ := 1
def friday_hours : ℕ := 3
def weeks_to_work : ℕ := 6

def weekly_hours : ℕ := monday_hours + wednesday_hours + friday_hours

def weekly_earnings : ℕ := weekly_hours * hourly_rate

theorem bicycle_cost : weekly_earnings * weeks_to_work = 180 := by
  sorry

end bicycle_cost_l1540_154088


namespace log_equation_solution_l1540_154090

/-- Given that log₃ₓ(343) = x and x is real, prove that x is a non-square, non-cube, non-integral rational number -/
theorem log_equation_solution (x : ℝ) (h : Real.log 343 / Real.log (3 * x) = x) :
  ∃ (a b : ℤ), x = (a : ℝ) / (b : ℝ) ∧ 
  b ≠ 0 ∧ 
  ¬ ∃ (n : ℤ), x = n ∧
  ¬ ∃ (n : ℝ), x = n ^ 2 ∧
  ¬ ∃ (n : ℝ), x = n ^ 3 :=
sorry

end log_equation_solution_l1540_154090


namespace smallest_y_for_square_l1540_154028

theorem smallest_y_for_square (y : ℕ) : y = 10 ↔ 
  (y > 0 ∧ 
   ∃ n : ℕ, 4410 * y = n^2 ∧
   ∀ z < y, z > 0 → ¬∃ m : ℕ, 4410 * z = m^2) := by
sorry

end smallest_y_for_square_l1540_154028


namespace interval_length_implies_c_minus_three_l1540_154093

theorem interval_length_implies_c_minus_three (c : ℝ) : 
  (∃ x : ℝ, 3 ≤ 5*x - 4 ∧ 5*x - 4 ≤ c) →
  (∀ x : ℝ, 3 ≤ 5*x - 4 ∧ 5*x - 4 ≤ c → (7/5 : ℝ) ≤ x ∧ x ≤ (c + 4)/5) →
  ((c + 4)/5 - 7/5 = 15) →
  c - 3 = 75 := by
sorry

end interval_length_implies_c_minus_three_l1540_154093


namespace ceiling_floor_difference_l1540_154045

theorem ceiling_floor_difference : 
  ⌈(15 / 8 : ℝ) * (-34 / 4 : ℝ)⌉ - ⌊(15 / 8 : ℝ) * ⌊-34 / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l1540_154045


namespace five_from_eight_l1540_154059

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem five_from_eight : choose 8 5 = 56 := by sorry

end five_from_eight_l1540_154059


namespace distribute_six_balls_three_boxes_l1540_154082

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes is 7 -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 7 := by sorry

end distribute_six_balls_three_boxes_l1540_154082


namespace congruence_implication_l1540_154064

theorem congruence_implication (a b c d n : ℤ) 
  (h1 : a * c ≡ 0 [ZMOD n])
  (h2 : b * c + a * d ≡ 0 [ZMOD n]) :
  b * c ≡ 0 [ZMOD n] ∧ a * d ≡ 0 [ZMOD n] := by
  sorry

end congruence_implication_l1540_154064


namespace min_value_theorem_l1540_154072

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m > 0) (h3 : n > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4) :=
by sorry

end min_value_theorem_l1540_154072


namespace sufficient_condition_implies_range_l1540_154004

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x : ℝ, |x - 1| < 3 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ |x - 1| ≥ 3) →
  a < -4 :=
by sorry

end sufficient_condition_implies_range_l1540_154004


namespace square_division_theorem_l1540_154020

/-- A type representing a square division -/
structure SquareDivision where
  n : ℕ
  is_valid : Bool

/-- Function that checks if a square can be divided into n smaller squares -/
def can_divide_square (n : ℕ) : Prop :=
  ∃ (sd : SquareDivision), sd.n = n ∧ sd.is_valid = true

theorem square_division_theorem :
  (∀ n : ℕ, n > 5 → can_divide_square n) ∧
  ¬(can_divide_square 2) ∧
  ¬(can_divide_square 3) := by sorry

end square_division_theorem_l1540_154020


namespace num_lions_seen_l1540_154050

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of chimps Borgnine has seen -/
def num_chimps : ℕ := 12

/-- The number of lizards Borgnine has seen -/
def num_lizards : ℕ := 5

/-- The number of tarantulas Borgnine will see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a chimp has -/
def chimp_legs : ℕ := 4

/-- The number of legs a lion has -/
def lion_legs : ℕ := 4

/-- The number of legs a lizard has -/
def lizard_legs : ℕ := 4

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

theorem num_lions_seen : ℕ := by
  sorry

end num_lions_seen_l1540_154050


namespace custom_mult_solution_l1540_154091

/-- Custom multiplication operation for integers -/
def customMult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if 21b = 160 under the custom multiplication, then b = 9 -/
theorem custom_mult_solution :
  ∀ b : ℤ, customMult 21 b = 160 → b = 9 := by
  sorry

end custom_mult_solution_l1540_154091


namespace liter_equals_cubic_decimeter_l1540_154010

-- Define the conversion factor between liters and cubic decimeters
def liter_to_cubic_decimeter : ℝ := 1

-- Theorem statement
theorem liter_equals_cubic_decimeter :
  1.5 * liter_to_cubic_decimeter = 1.5 := by sorry

end liter_equals_cubic_decimeter_l1540_154010


namespace cookie_box_cost_josh_cookie_box_cost_l1540_154083

/-- The cost of a box of cookies given Josh's bracelet-making business --/
theorem cookie_box_cost (cost_per_bracelet : ℚ) (price_per_bracelet : ℚ) 
  (num_bracelets : ℕ) (money_left : ℚ) : ℚ :=
  let profit_per_bracelet := price_per_bracelet - cost_per_bracelet
  let total_profit := profit_per_bracelet * num_bracelets
  total_profit - money_left

/-- The cost of Josh's box of cookies is $3 --/
theorem josh_cookie_box_cost : cookie_box_cost 1 1.5 12 3 = 3 := by
  sorry

end cookie_box_cost_josh_cookie_box_cost_l1540_154083


namespace top_tier_lamps_l1540_154041

/-- Represents the number of tiers in the tower -/
def n : ℕ := 7

/-- Represents the common ratio of the geometric sequence -/
def r : ℕ := 2

/-- Represents the total number of lamps in the tower -/
def total_lamps : ℕ := 381

/-- Calculates the sum of a geometric series -/
def geometric_sum (a₁ : ℕ) : ℕ := a₁ * (1 - r^n) / (1 - r)

/-- Theorem stating that the number of lamps on the top tier is 3 -/
theorem top_tier_lamps : ∃ (a₁ : ℕ), geometric_sum a₁ = total_lamps ∧ a₁ = 3 := by
  sorry

end top_tier_lamps_l1540_154041


namespace solution_set_implies_sum_l1540_154005

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x + 2

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ)) →
  a + b = -14 := by
  sorry

end solution_set_implies_sum_l1540_154005


namespace bookstore_sales_after_returns_l1540_154065

/-- Calculates the total sales after returns for a bookstore --/
theorem bookstore_sales_after_returns 
  (total_customers : ℕ) 
  (return_rate : ℚ) 
  (price_per_book : ℕ) : 
  total_customers = 1000 → 
  return_rate = 37 / 100 → 
  price_per_book = 15 → 
  (total_customers : ℚ) * (1 - return_rate) * (price_per_book : ℚ) = 9450 := by
  sorry

end bookstore_sales_after_returns_l1540_154065


namespace divisible_by_five_l1540_154074

theorem divisible_by_five (a b : ℕ) : 
  (5 ∣ a * b) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end divisible_by_five_l1540_154074


namespace exponent_multiplication_l1540_154034

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l1540_154034


namespace patio_rearrangement_l1540_154032

theorem patio_rearrangement (total_tiles : ℕ) (initial_rows : ℕ) (added_rows : ℕ) :
  total_tiles = 126 →
  initial_rows = 9 →
  added_rows = 4 →
  ∃ (initial_columns final_columns : ℕ),
    initial_columns * initial_rows = total_tiles ∧
    final_columns * (initial_rows + added_rows) = total_tiles ∧
    initial_columns - final_columns = 5 :=
by sorry

end patio_rearrangement_l1540_154032


namespace power_sum_equality_l1540_154039

theorem power_sum_equality (x y a b : ℝ) (h1 : x + y = a + b) (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n := by
  sorry

end power_sum_equality_l1540_154039


namespace perpendicular_line_angle_l1540_154057

-- Define the perpendicularity condition
def isPerpendicular (θ : Real) : Prop :=
  ∃ t : Real, (1 + t * Real.cos θ = t * Real.sin θ) ∧ 
              (Real.tan θ = -1)

-- State the theorem
theorem perpendicular_line_angle :
  ∀ θ : Real, 0 ≤ θ ∧ θ < π → isPerpendicular θ → θ = 3 * π / 4 := by
  sorry

end perpendicular_line_angle_l1540_154057


namespace divisibility_by_seven_l1540_154086

theorem divisibility_by_seven (n : ℕ) : 7 ∣ (3^(12*n + 1) + 2^(6*n + 2)) :=
sorry

end divisibility_by_seven_l1540_154086


namespace crayon_box_count_l1540_154085

theorem crayon_box_count : ∀ (total : ℕ),
  (total : ℚ) / 3 + (total : ℚ) * (1 / 5) + 56 = total →
  total = 120 := by
sorry

end crayon_box_count_l1540_154085


namespace intersection_of_A_and_B_l1540_154048

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by sorry

end intersection_of_A_and_B_l1540_154048


namespace min_resistance_optimal_l1540_154043

noncomputable def min_resistance (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ :=
  let r₁₂ := (a₁ * a₂) / (a₁ + a₂)
  let r₁₂₃ := r₁₂ + a₃
  let r₄₅ := (a₄ * a₅) / (a₄ + a₅)
  let r₄₅₆ := r₄₅ + a₆
  (r₁₂₃ * r₄₅₆) / (r₁₂₃ + r₄₅₆)

theorem min_resistance_optimal
  (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅ ∧ a₅ > a₆) :
  ∀ (r : ℝ), r ≥ min_resistance a₁ a₂ a₃ a₄ a₅ a₆ :=
by sorry

end min_resistance_optimal_l1540_154043


namespace factorial_sum_square_solutions_l1540_154069

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_square_solutions :
  ∀ m n : ℕ+, m^2 = factorial_sum n ↔ (m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 3) := by
  sorry

end factorial_sum_square_solutions_l1540_154069


namespace log_sum_equation_l1540_154025

theorem log_sum_equation (k : ℤ) (x : ℝ) 
  (h : (7.318 * Real.log x / Real.log k) + 
       (Real.log x / Real.log (k ^ (1/2 : ℝ))) + 
       (Real.log x / Real.log (k ^ (1/3 : ℝ))) + 
       -- ... (representing the sum up to k terms)
       (Real.log x / Real.log (k ^ (1/k : ℝ))) = 
       (k + 1 : ℝ) / 2) :
  x = k ^ (1/k : ℝ) := by
sorry

end log_sum_equation_l1540_154025


namespace counterexample_exists_l1540_154019

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a⁻¹ ≥ b⁻¹ := by
  sorry

end counterexample_exists_l1540_154019


namespace negation_of_existential_proposition_negation_of_specific_proposition_l1540_154054

theorem negation_of_existential_proposition (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end negation_of_existential_proposition_negation_of_specific_proposition_l1540_154054


namespace employee_devices_l1540_154058

theorem employee_devices (total : ℝ) (h_total : total > 0) : 
  let cell_phone := (2/3 : ℝ) * total
  let pager := (2/5 : ℝ) * total
  let neither := (1/3 : ℝ) * total
  let both := cell_phone + pager - (total - neither)
  both / total = 2/5 := by
sorry

end employee_devices_l1540_154058


namespace suitcase_lock_settings_l1540_154087

/-- The number of dials on the suitcase lock. -/
def num_dials : ℕ := 4

/-- The number of digits available for each dial. -/
def num_digits : ℕ := 10

/-- Calculates the number of different settings for a suitcase lock. -/
def lock_settings : ℕ := num_digits * (num_digits - 1) * (num_digits - 2) * (num_digits - 3)

/-- Theorem stating that the number of different settings for the suitcase lock is 5040. -/
theorem suitcase_lock_settings :
  lock_settings = 5040 := by sorry

end suitcase_lock_settings_l1540_154087


namespace fertilizer_pesticide_cost_l1540_154061

/-- Proves the amount spent on fertilizers and pesticides for a small farm operation --/
theorem fertilizer_pesticide_cost
  (seed_cost : ℝ)
  (labor_cost : ℝ)
  (num_bags : ℕ)
  (price_per_bag : ℝ)
  (profit_percentage : ℝ)
  (h1 : seed_cost = 50)
  (h2 : labor_cost = 15)
  (h3 : num_bags = 10)
  (h4 : price_per_bag = 11)
  (h5 : profit_percentage = 0.1)
  : ∃ (fertilizer_pesticide_cost : ℝ),
    fertilizer_pesticide_cost = 35 ∧
    price_per_bag * num_bags = (1 + profit_percentage) * (seed_cost + labor_cost + fertilizer_pesticide_cost) :=
by sorry

end fertilizer_pesticide_cost_l1540_154061


namespace hexagon_extension_length_l1540_154053

-- Define the regular hexagon
def RegularHexagon (C D E F G H : ℝ × ℝ) : Prop :=
  -- Add conditions for a regular hexagon with side length 4
  sorry

-- Define the extension of CD to Y
def ExtendCD (C D Y : ℝ × ℝ) : Prop :=
  dist C Y = 2 * dist C D

-- Main theorem
theorem hexagon_extension_length 
  (C D E F G H Y : ℝ × ℝ) 
  (hex : RegularHexagon C D E F G H) 
  (ext : ExtendCD C D Y) : 
  dist H Y = 6 * Real.sqrt 5 := by
  sorry

end hexagon_extension_length_l1540_154053


namespace triangle_toothpicks_count_l1540_154084

/-- The number of small triangles in the base of the large triangle -/
def base_triangles : ℕ := 101

/-- The total number of small triangles in the large triangle -/
def total_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of shared toothpicks in the structure -/
def shared_toothpicks (n : ℕ) : ℕ := 3 * total_triangles n / 2

/-- The number of boundary toothpicks -/
def boundary_toothpicks (n : ℕ) : ℕ := 3 * n

/-- The number of support toothpicks on the boundary -/
def support_toothpicks : ℕ := 3

/-- The total number of toothpicks required for the structure -/
def total_toothpicks (n : ℕ) : ℕ :=
  shared_toothpicks n + boundary_toothpicks n + support_toothpicks

theorem triangle_toothpicks_count :
  total_toothpicks base_triangles = 8032 :=
sorry

end triangle_toothpicks_count_l1540_154084


namespace apple_juice_quantity_l1540_154094

/-- Given the total apple production and export percentage, calculate the quantity of apples used for juice -/
theorem apple_juice_quantity (total_production : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : 
  total_production = 6 →
  export_percentage = 0.25 →
  juice_percentage = 0.60 →
  juice_percentage * (total_production * (1 - export_percentage)) = 2.7 := by
sorry

end apple_juice_quantity_l1540_154094


namespace video_dislikes_l1540_154000

/-- Represents the number of dislikes for a video -/
def dislikes (likes : ℕ) (additional : ℕ) (extra : ℕ) : ℕ :=
  likes / 2 + additional + extra

/-- Theorem stating the final number of dislikes for the video -/
theorem video_dislikes : dislikes 3000 100 1000 = 2600 := by
  sorry

end video_dislikes_l1540_154000


namespace base_b_square_l1540_154033

theorem base_b_square (b : ℕ) : 
  (2 * b + 4)^2 = 5 * b^2 + 5 * b + 4 → b = 12 := by
  sorry

end base_b_square_l1540_154033


namespace spears_from_sapling_proof_l1540_154067

/-- The number of spears that can be made from a log -/
def spears_per_log : ℕ := 9

/-- The number of spears that can be made from 6 saplings and a log -/
def spears_from_6_saplings_and_log : ℕ := 27

/-- The number of saplings used along with a log -/
def number_of_saplings : ℕ := 6

/-- The number of spears that can be made from a single sapling -/
def spears_per_sapling : ℕ := 3

theorem spears_from_sapling_proof :
  number_of_saplings * spears_per_sapling + spears_per_log = spears_from_6_saplings_and_log :=
by sorry

end spears_from_sapling_proof_l1540_154067


namespace water_conservation_function_correct_water_conserved_third_year_correct_l1540_154035

/-- Represents the water conservation model for a city's tree planting program. -/
structure WaterConservationModel where
  /-- The number of trees planted annually (in millions) -/
  annual_trees : ℕ
  /-- The initial water conservation in 2009 (in million cubic meters) -/
  initial_conservation : ℕ
  /-- The water conservation by 2015 (in billion cubic meters) -/
  final_conservation : ℚ
  /-- The year considered as the first year -/
  start_year : ℕ
  /-- The year when the forest city construction will be completed -/
  end_year : ℕ

/-- The water conservation function for the city's tree planting program. -/
def water_conservation_function (model : WaterConservationModel) (x : ℚ) : ℚ :=
  (4/3) * x + (5/3)

/-- Theorem stating that the given water conservation function is correct for the model. -/
theorem water_conservation_function_correct (model : WaterConservationModel) 
  (h1 : model.annual_trees = 500)
  (h2 : model.initial_conservation = 300)
  (h3 : model.final_conservation = 11/10)
  (h4 : model.start_year = 2009)
  (h5 : model.end_year = 2015) :
  ∀ x : ℚ, 1 ≤ x ∧ x ≤ 7 →
    water_conservation_function model x = 
      (model.final_conservation - (model.initial_conservation / 1000)) / (model.end_year - model.start_year) * x + 
      (model.initial_conservation / 1000) := by
  sorry

/-- Theorem stating that the water conserved in the third year (2011) is correct. -/
theorem water_conserved_third_year_correct (model : WaterConservationModel) :
  water_conservation_function model 3 = 17/3 := by
  sorry

end water_conservation_function_correct_water_conserved_third_year_correct_l1540_154035


namespace ham_and_cake_probability_l1540_154009

/-- The probability of packing a ham sandwich and cake on the same day -/
theorem ham_and_cake_probability :
  let total_days : ℕ := 5
  let ham_days : ℕ := 3
  let cake_days : ℕ := 1
  (ham_days : ℚ) / total_days * cake_days / total_days = 3 / 25 := by
  sorry

end ham_and_cake_probability_l1540_154009


namespace min_value_xy_over_x2_plus_y2_l1540_154078

theorem min_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) : 
  x * y / (x^2 + y^2) ≥ 0.4 := by
  sorry

end min_value_xy_over_x2_plus_y2_l1540_154078


namespace cool_parents_problem_l1540_154029

theorem cool_parents_problem (U : Finset ℕ) (A B : Finset ℕ) 
  (h1 : Finset.card U = 40)
  (h2 : Finset.card A = 18)
  (h3 : Finset.card B = 20)
  (h4 : Finset.card (A ∩ B) = 11)
  (h5 : A ⊆ U)
  (h6 : B ⊆ U) :
  Finset.card (U \ (A ∪ B)) = 13 := by
  sorry

end cool_parents_problem_l1540_154029


namespace min_face_sum_l1540_154098

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 8

/-- The sum of any three numbers on the same face is at least 10 -/
def ValidArrangement (arr : CubeArrangement) : Prop :=
  ∀ (face : Fin 6) (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 →
    (arr (face * 4 + v1) + arr (face * 4 + v2) + arr (face * 4 + v3) : ℕ) ≥ 10

/-- The sum of numbers on one face -/
def FaceSum (arr : CubeArrangement) (face : Fin 6) : ℕ :=
  (arr (face * 4) : ℕ) + (arr (face * 4 + 1) : ℕ) + (arr (face * 4 + 2) : ℕ) + (arr (face * 4 + 3) : ℕ)

/-- The minimum possible sum of numbers on one face is 16 -/
theorem min_face_sum :
  ∀ (arr : CubeArrangement), ValidArrangement arr →
    ∃ (face : Fin 6), FaceSum arr face = 16 ∧
      ∀ (other_face : Fin 6), FaceSum arr other_face ≥ 16 :=
sorry

end min_face_sum_l1540_154098


namespace cube_preserves_inequality_l1540_154008

theorem cube_preserves_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_preserves_inequality_l1540_154008


namespace arithmetic_sequence_sum_l1540_154013

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 3rd to 7th terms equals 450. -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SumCondition a → a 2 + a 8 = 180 := by
  sorry

end arithmetic_sequence_sum_l1540_154013


namespace special_polynomial_at_zero_l1540_154026

/-- A polynomial of degree 6 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, ∀ x, p x = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) ∧
  (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (2^n))

/-- Theorem stating that a special polynomial evaluates to 0 at x = 0 -/
theorem special_polynomial_at_zero
  (p : ℝ → ℝ)
  (h : special_polynomial p) :
  p 0 = 0 := by
  sorry

end special_polynomial_at_zero_l1540_154026


namespace square_of_negative_sqrt_two_l1540_154079

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by sorry

end square_of_negative_sqrt_two_l1540_154079


namespace star_op_equation_solution_l1540_154056

-- Define the "※" operation
def star_op (a b : ℝ) : ℝ := a * b^2 + 2 * a * b

-- State the theorem
theorem star_op_equation_solution :
  ∃! x : ℝ, star_op 1 x = -1 ∧ x = -1 := by
  sorry

end star_op_equation_solution_l1540_154056


namespace no_extrema_iff_a_nonpositive_l1540_154081

/-- The function f(x) = x^2 - a * ln(x) has no extrema if and only if a ≤ 0 -/
theorem no_extrema_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y > 0 ∧ (x^2 - a * Real.log x < y^2 - a * Real.log y ∨ 
                                      x^2 - a * Real.log x > y^2 - a * Real.log y)) ↔ 
  a ≤ 0 :=
sorry

end no_extrema_iff_a_nonpositive_l1540_154081


namespace largest_when_first_changed_l1540_154022

def original : ℚ := 0.12345

def change_digit (n : ℕ) : ℚ :=
  match n with
  | 1 => 0.92345
  | 2 => 0.19345
  | 3 => 0.12945
  | 4 => 0.12395
  | 5 => 0.12349
  | _ => original

theorem largest_when_first_changed :
  ∀ n : ℕ, n ≥ 1 → n ≤ 5 → change_digit 1 ≥ change_digit n :=
sorry

end largest_when_first_changed_l1540_154022


namespace max_subway_commuters_l1540_154055

theorem max_subway_commuters (total_employees : ℕ) 
  (h_total : total_employees = 48) 
  (part_time full_time : ℕ) 
  (h_sum : part_time + full_time = total_employees) 
  (h_both_exist : part_time > 0 ∧ full_time > 0) :
  ∃ (subway_commuters : ℕ), 
    subway_commuters = ⌊(1 / 3 : ℚ) * part_time⌋ + ⌊(1 / 4 : ℚ) * full_time⌋ ∧
    subway_commuters ≤ 15 ∧
    (∀ (pt ft : ℕ), 
      pt + ft = total_employees → 
      pt > 0 → 
      ft > 0 → 
      ⌊(1 / 3 : ℚ) * pt⌋ + ⌊(1 / 4 : ℚ) * ft⌋ ≤ subway_commuters) :=
by sorry

end max_subway_commuters_l1540_154055


namespace G_fraction_is_lowest_terms_denominator_minus_numerator_l1540_154014

/-- G is defined as the infinite repeating decimal 0.837837837... -/
def G : ℚ := 837 / 999

/-- The fraction representation of G in lowest terms -/
def G_fraction : ℚ := 31 / 37

theorem G_fraction_is_lowest_terms : G = G_fraction := by sorry

theorem denominator_minus_numerator : Nat.gcd 31 37 = 1 ∧ 37 - 31 = 6 := by sorry

end G_fraction_is_lowest_terms_denominator_minus_numerator_l1540_154014


namespace eight_power_91_greater_than_seven_power_92_l1540_154071

theorem eight_power_91_greater_than_seven_power_92 : 8^91 > 7^92 := by
  sorry

end eight_power_91_greater_than_seven_power_92_l1540_154071


namespace inequality_solution_set_l1540_154066

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 := by
  sorry

end inequality_solution_set_l1540_154066


namespace reflection_result_l1540_154062

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -(p.2))

/-- The original point F -/
def F : ℝ × ℝ := (3, -3)

theorem reflection_result :
  (reflect_x (reflect_y F)) = (-3, 3) := by sorry

end reflection_result_l1540_154062


namespace solution_set_l1540_154052

theorem solution_set : 
  {x : ℝ | x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3*x + 2 > 0} = {-3, 0, 5} := by
  sorry

end solution_set_l1540_154052


namespace hillarys_remaining_money_l1540_154017

/-- Calculates the amount Hillary is left with after selling crafts and accounting for all costs and transactions. -/
theorem hillarys_remaining_money
  (base_price : ℝ)
  (cost_per_craft : ℝ)
  (crafts_sold : ℕ)
  (extra_money : ℝ)
  (tax_rate : ℝ)
  (deposit_amount : ℝ)
  (h1 : base_price = 12)
  (h2 : cost_per_craft = 4)
  (h3 : crafts_sold = 3)
  (h4 : extra_money = 7)
  (h5 : tax_rate = 0.1)
  (h6 : deposit_amount = 26)
  : ∃ (remaining : ℝ), remaining = 1.9 ∧ remaining ≥ 0 := by
  sorry

#check hillarys_remaining_money

end hillarys_remaining_money_l1540_154017


namespace spheres_in_unit_cube_radius_l1540_154073

/-- A configuration of spheres in a unit cube -/
structure SpheresInCube where
  /-- The number of spheres in the cube -/
  num_spheres : ℕ
  /-- The radius of each sphere -/
  radius : ℝ
  /-- One sphere is at a vertex of the cube -/
  vertex_sphere : Prop
  /-- Each of the remaining spheres is tangent to the vertex sphere and three faces of the cube -/
  remaining_spheres_tangent : Prop

/-- The theorem stating the radius of spheres in the given configuration -/
theorem spheres_in_unit_cube_radius (config : SpheresInCube) :
  config.num_spheres = 12 →
  config.vertex_sphere →
  config.remaining_spheres_tangent →
  config.radius = 1 / 2 := by
  sorry


end spheres_in_unit_cube_radius_l1540_154073


namespace f_theorem_l1540_154099

def f_properties (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  (∀ x, f (-x) = f x) ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  f (-1) = 0

theorem f_theorem (f : ℝ → ℝ) (h : f_properties f) :
  f 3 > f 4 ∧
  (∀ m, f (m - 1) < f 2 → m < -1 ∨ m > 3) ∧
  ∃ M, ∀ x, f x ≤ M :=
sorry

end f_theorem_l1540_154099


namespace inequality_proof_l1540_154012

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 3) / 2 :=
by sorry

end inequality_proof_l1540_154012


namespace solve_system_l1540_154070

theorem solve_system (x y : ℝ) (eq1 : x - y = 8) (eq2 : x + 2*y = 14) : x = 10 := by
  sorry

end solve_system_l1540_154070


namespace expression_simplification_l1540_154068

theorem expression_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3*x^3 - 6*x^2 + 7*x + 2 := by
  sorry

end expression_simplification_l1540_154068


namespace train_passing_pole_l1540_154027

theorem train_passing_pole (train_length platform_length : ℝ) 
  (platform_passing_time : ℝ) : 
  train_length = 120 →
  platform_length = 120 →
  platform_passing_time = 22 →
  (∃ (pole_passing_time : ℝ), 
    pole_passing_time = train_length / (train_length + platform_length) * platform_passing_time ∧
    pole_passing_time = 11) :=
by sorry

end train_passing_pole_l1540_154027


namespace ellipse_semi_minor_axis_l1540_154003

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis, 
    prove that its semi-minor axis has length √7 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (2, -1))
  (h_focus : focus = (2, -4))
  (h_semi_major_endpoint : semi_major_endpoint = (2, 3)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 7 := by
  sorry

end ellipse_semi_minor_axis_l1540_154003


namespace fudge_pan_dimensions_l1540_154031

theorem fudge_pan_dimensions (side1 : ℝ) (area : ℝ) : 
  side1 = 29 → area = 522 → (area / side1) = 18 := by
  sorry

end fudge_pan_dimensions_l1540_154031


namespace initial_marbles_calculation_l1540_154023

theorem initial_marbles_calculation (a b : ℚ) :
  a/b + 489.35 = 2778.65 →
  a/b = 2289.3 := by
sorry

end initial_marbles_calculation_l1540_154023


namespace sum_of_reciprocals_l1540_154016

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = (x * y)^2) : 
  1 / x + 1 / y = 1 := by
sorry

end sum_of_reciprocals_l1540_154016


namespace half_angle_quadrant_l1540_154030

theorem half_angle_quadrant (α : Real) : 
  (π / 2 < α ∧ α < π) → 
  ((0 < (α / 2) ∧ (α / 2) < π / 2) ∨ (π < (α / 2) ∧ (α / 2) < 3 * π / 2)) :=
by sorry

end half_angle_quadrant_l1540_154030


namespace max_a_value_l1540_154051

theorem max_a_value (a k x₁ x₂ : ℝ) : 
  (∀ k ∈ Set.Icc 0 2, 
   ∀ x₁ ∈ Set.Icc k (k + a), 
   ∀ x₂ ∈ Set.Icc (k + 2*a) (k + 4*a), 
   (x₁^2 - (k^2 - 5*a*k + 3)*x₁ + 7) ≥ (x₂^2 - (k^2 - 5*a*k + 3)*x₂ + 7)) →
  a ≤ (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end max_a_value_l1540_154051


namespace power_of_three_mod_eleven_l1540_154095

theorem power_of_three_mod_eleven : 3^2023 % 11 = 5 := by
  sorry

end power_of_three_mod_eleven_l1540_154095


namespace trapezoid_sides_for_given_circle_l1540_154037

/-- Represents a trapezoid formed by tangents to a circle -/
structure CircleTrapezoid where
  radius : ℝ
  chord_length : ℝ

/-- Calculates the sides of the trapezoid -/
def trapezoid_sides (t : CircleTrapezoid) : (ℝ × ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct sides of the trapezoid for the given circle -/
theorem trapezoid_sides_for_given_circle :
  let t : CircleTrapezoid := { radius := 5, chord_length := 8 }
  trapezoid_sides t = (12.5, 5, 12.5, 20) := by
  sorry

end trapezoid_sides_for_given_circle_l1540_154037


namespace marks_deck_cost_l1540_154060

/-- Calculates the total cost of constructing and sealing a rectangular deck. -/
def deck_total_cost (length width construction_cost_per_sqft sealant_cost_per_sqft : ℝ) : ℝ :=
  let area := length * width
  let construction_cost := area * construction_cost_per_sqft
  let sealant_cost := area * sealant_cost_per_sqft
  construction_cost + sealant_cost

/-- Theorem stating that the total cost of Mark's deck is $4800. -/
theorem marks_deck_cost : 
  deck_total_cost 30 40 3 1 = 4800 := by
  sorry

end marks_deck_cost_l1540_154060


namespace calculator_battery_life_l1540_154047

/-- Calculates the remaining battery life of a calculator after partial use and an exam -/
theorem calculator_battery_life 
  (full_battery : ℝ) 
  (used_fraction : ℝ) 
  (exam_duration : ℝ) 
  (h1 : full_battery = 60) 
  (h2 : used_fraction = 3/4) 
  (h3 : exam_duration = 2) :
  full_battery * (1 - used_fraction) - exam_duration = 13 := by
  sorry

end calculator_battery_life_l1540_154047


namespace smallest_c_for_inverse_l1540_154063

def g (x : ℝ) : ℝ := (x + 3)^2 - 10

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y : ℝ, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ -3 :=
sorry

end smallest_c_for_inverse_l1540_154063


namespace intersection_M_N_l1540_154001

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_M_N_l1540_154001


namespace eden_has_fourteen_bears_l1540_154089

/-- The number of stuffed bears Eden has after receiving her share from Daragh --/
def edens_final_bear_count (initial_bears : ℕ) (favorite_bears : ℕ) (sisters : ℕ) (edens_initial_bears : ℕ) : ℕ :=
  let remaining_bears := initial_bears - favorite_bears
  let bears_per_sister := remaining_bears / sisters
  edens_initial_bears + bears_per_sister

/-- Theorem stating that Eden will have 14 stuffed bears after receiving her share --/
theorem eden_has_fourteen_bears :
  edens_final_bear_count 20 8 3 10 = 14 := by
  sorry

end eden_has_fourteen_bears_l1540_154089


namespace bluetooth_module_stock_l1540_154042

theorem bluetooth_module_stock (total_modules : ℕ) (total_cost : ℚ)
  (expensive_cost cheap_cost : ℚ) :
  total_modules = 11 →
  total_cost = 45 →
  expensive_cost = 10 →
  cheap_cost = 7/2 →
  ∃ (expensive_count cheap_count : ℕ),
    expensive_count + cheap_count = total_modules ∧
    expensive_count * expensive_cost + cheap_count * cheap_cost = total_cost ∧
    cheap_count = 10 := by
  sorry

end bluetooth_module_stock_l1540_154042


namespace arithmetic_sequence_sum_16_l1540_154038

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term_eq : a 1 = a 1  -- Placeholder for the first term
  diff_eq : ∀ n, a (n + 1) - a n = d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_16 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 12 = -8) 
  (h2 : sum_n seq 9 = -9) : 
  sum_n seq 16 = -72 := by
  sorry

end arithmetic_sequence_sum_16_l1540_154038


namespace money_ratio_l1540_154097

/-- Jake's feeding allowance in dollars -/
def feeding_allowance : ℚ := 4

/-- Cost of one candy in dollars -/
def candy_cost : ℚ := 1/5

/-- Number of candies Jake's friend can purchase -/
def candies_purchased : ℕ := 5

/-- Amount of money Jake gave to his friend in dollars -/
def money_given : ℚ := candy_cost * candies_purchased

/-- Theorem stating the ratio of money given to feeding allowance -/
theorem money_ratio : money_given / feeding_allowance = 1/4 := by
  sorry

end money_ratio_l1540_154097


namespace total_age_proof_l1540_154096

/-- Given three people a, b, and c, where a is two years older than b, b is twice as old as c, 
    and b is 10 years old, prove that the total of their ages is 27 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 10 → a = b + 2 → b = 2 * c → a + b + c = 27 := by
  sorry

end total_age_proof_l1540_154096


namespace cos_330_degrees_l1540_154075

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l1540_154075


namespace equation_solution_l1540_154046

theorem equation_solution :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 4/5 →
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 →
  x = -4/3 :=
by
  sorry

end equation_solution_l1540_154046


namespace decimal_between_four_and_five_l1540_154006

theorem decimal_between_four_and_five : ∃ x : ℝ, (x = 4.5) ∧ (4 < x) ∧ (x < 5) := by
  sorry

end decimal_between_four_and_five_l1540_154006


namespace distance_between_complex_points_l1540_154036

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 + 3 * Complex.I
  let z₂ : ℂ := -2 + Real.sqrt 2 * Complex.I
  Complex.abs (z₁ - z₂) = Real.sqrt (36 - 6 * Real.sqrt 2) :=
by sorry

end distance_between_complex_points_l1540_154036


namespace quadratic_equation_1_l1540_154024

theorem quadratic_equation_1 : ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 3 ∧
  x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0 := by
  sorry

end quadratic_equation_1_l1540_154024


namespace bicycles_in_garage_l1540_154092

/-- The number of bicycles in Connor's garage --/
def num_bicycles : ℕ := 20

/-- The number of cars in Connor's garage --/
def num_cars : ℕ := 10

/-- The number of motorcycles in Connor's garage --/
def num_motorcycles : ℕ := 5

/-- The total number of wheels in Connor's garage --/
def total_wheels : ℕ := 90

/-- The number of wheels on a bicycle --/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a car --/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a motorcycle --/
def wheels_per_motorcycle : ℕ := 2

theorem bicycles_in_garage :
  num_bicycles * wheels_per_bicycle +
  num_cars * wheels_per_car +
  num_motorcycles * wheels_per_motorcycle = total_wheels :=
by sorry

end bicycles_in_garage_l1540_154092


namespace inverse_inequality_for_negative_reals_l1540_154076

theorem inverse_inequality_for_negative_reals (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  1 / a > 1 / b := by
sorry

end inverse_inequality_for_negative_reals_l1540_154076


namespace GH_distance_is_40_l1540_154011

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of a diagonal -/
  diagonal_length : ℝ
  /-- The distance from point G to vertex A -/
  GA_distance : ℝ
  /-- The distance from point G to vertex D -/
  GD_distance : ℝ
  /-- The base angle at the longer base (AD) -/
  base_angle : ℝ
  /-- Assumption that the diagonal length is 20√5 -/
  diagonal_length_eq : diagonal_length = 20 * Real.sqrt 5
  /-- Assumption that GA distance is 20 -/
  GA_distance_eq : GA_distance = 20
  /-- Assumption that GD distance is 40 -/
  GD_distance_eq : GD_distance = 40
  /-- Assumption that the base angle is π/4 -/
  base_angle_eq : base_angle = Real.pi / 4

/-- The main theorem stating that GH distance is 40 -/
theorem GH_distance_is_40 (t : IsoscelesTrapezoid) : ℝ := by
  sorry

#check GH_distance_is_40

end GH_distance_is_40_l1540_154011


namespace pizza_combinations_l1540_154044

theorem pizza_combinations (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end pizza_combinations_l1540_154044


namespace n_value_l1540_154002

theorem n_value : ∃ (n : ℤ), (1/6 : ℚ) < (n : ℚ)/24 ∧ (n : ℚ)/24 < (1/4 : ℚ) → n = 5 := by
  sorry

end n_value_l1540_154002


namespace second_term_of_geometric_series_l1540_154040

/-- For an infinite geometric series with common ratio 1/4 and sum 16, the second term is 3 -/
theorem second_term_of_geometric_series : 
  ∀ (a : ℝ), 
  (a / (1 - (1/4 : ℝ)) = 16) →  -- Sum of infinite geometric series
  (a * (1/4 : ℝ) = 3) :=        -- Second term
by sorry

end second_term_of_geometric_series_l1540_154040


namespace solve_exponential_equation_l1540_154007

theorem solve_exponential_equation :
  ∃ x : ℝ, 3^(2*x + 1) = (1/81 : ℝ) ∧ x = -5/2 := by
  sorry

end solve_exponential_equation_l1540_154007


namespace fence_sections_count_l1540_154080

/-- The number of posts in the nth section -/
def posts_in_section (n : ℕ) : ℕ := 2 * n + 1

/-- The total number of posts used for n sections -/
def total_posts (n : ℕ) : ℕ := n^2

/-- The total number of posts available -/
def available_posts : ℕ := 435

theorem fence_sections_count :
  ∃ (n : ℕ), total_posts n = available_posts ∧ n = 21 :=
sorry

end fence_sections_count_l1540_154080
