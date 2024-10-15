import Mathlib

namespace NUMINAMATH_GPT_find_product_xyz_l1319_131951

-- Definitions for the given conditions
variables (x y z : ℕ) -- positive integers

-- Conditions
def condition1 : Prop := x + 2 * y = z
def condition2 : Prop := x^2 - 4 * y^2 + z^2 = 310

-- Theorem statement
theorem find_product_xyz (h1 : condition1 x y z) (h2 : condition2 x y z) : 
  x * y * z = 11935 ∨ x * y * z = 2015 :=
sorry

end NUMINAMATH_GPT_find_product_xyz_l1319_131951


namespace NUMINAMATH_GPT_find_height_of_larger_cuboid_l1319_131936

-- Define the larger cuboid dimensions
def Length_large : ℝ := 18
def Width_large : ℝ := 15
def Volume_large (Height_large : ℝ) : ℝ := Length_large * Width_large * Height_large

-- Define the smaller cuboid dimensions
def Length_small : ℝ := 5
def Width_small : ℝ := 6
def Height_small : ℝ := 3
def Volume_small : ℝ := Length_small * Width_small * Height_small

-- Define the total volume of 6 smaller cuboids
def Total_volume_small : ℝ := 6 * Volume_small

-- State the problem and the proof goal
theorem find_height_of_larger_cuboid : 
  ∃ H : ℝ, Volume_large H = Total_volume_small :=
by
  use 2
  sorry

end NUMINAMATH_GPT_find_height_of_larger_cuboid_l1319_131936


namespace NUMINAMATH_GPT_gambler_largest_amount_received_l1319_131935

def largest_amount_received_back (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : ℕ :=
  3000 - (30 * a + 100 * b)

theorem gambler_largest_amount_received (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : 
    largest_amount_received_back x y a b h1 h2 h3 = 2030 :=
by sorry

end NUMINAMATH_GPT_gambler_largest_amount_received_l1319_131935


namespace NUMINAMATH_GPT_quadratic_equal_roots_k_value_l1319_131943

theorem quadratic_equal_roots_k_value (k : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 4 * k = 0 → x^2 - 8 * x - 4 * k = 0 ∧ (0 : ℝ) = 0 ) →
  k = -4 :=
sorry

end NUMINAMATH_GPT_quadratic_equal_roots_k_value_l1319_131943


namespace NUMINAMATH_GPT_cherries_purchase_l1319_131999

theorem cherries_purchase (total_money : ℝ) (price_per_kg : ℝ) 
  (genevieve_money : ℝ) (shortage : ℝ) (clarice_money : ℝ) :
  genevieve_money = 1600 → shortage = 400 → clarice_money = 400 → price_per_kg = 8 →
  total_money = genevieve_money + shortage + clarice_money →
  total_money / price_per_kg = 250 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_cherries_purchase_l1319_131999


namespace NUMINAMATH_GPT_share_per_person_in_dollars_l1319_131908

-- Definitions based on conditions
def total_cost_euros : ℝ := 25 * 10^9  -- 25 billion Euros
def number_of_people : ℝ := 300 * 10^6  -- 300 million people
def exchange_rate : ℝ := 1.2  -- 1 Euro = 1.2 dollars

-- To prove
theorem share_per_person_in_dollars : (total_cost_euros * exchange_rate) / number_of_people = 100 := 
by 
  sorry

end NUMINAMATH_GPT_share_per_person_in_dollars_l1319_131908


namespace NUMINAMATH_GPT_sum_of_squares_of_ages_l1319_131949

theorem sum_of_squares_of_ages 
  (d t h : ℕ) 
  (cond1 : 3 * d + t = 2 * h)
  (cond2 : 2 * h ^ 3 = 3 * d ^ 3 + t ^ 3)
  (rel_prime : Nat.gcd d (Nat.gcd t h) = 1) :
  d ^ 2 + t ^ 2 + h ^ 2 = 42 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_ages_l1319_131949


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l1319_131993

theorem common_ratio_geometric_sequence (a₃ S₃ : ℝ) (q : ℝ)
  (h1 : a₃ = 7) (h2 : S₃ = 21)
  (h3 : ∃ a₁ : ℝ, a₃ = a₁ * q^2)
  (h4 : ∃ a₁ : ℝ, S₃ = a₁ * (1 + q + q^2)) :
  q = -1/2 ∨ q = 1 :=
sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l1319_131993


namespace NUMINAMATH_GPT_solution_set_l1319_131945

theorem solution_set (x : ℝ) : (x + 1 = |x + 3| - |x - 1|) ↔ (x = 3 ∨ x = -1 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1319_131945


namespace NUMINAMATH_GPT_range_of_y_l1319_131915

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.arcsin x

theorem range_of_y : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 →
  y x ∈ Set.Icc (-Real.sin 1 - (Real.pi / 2)) (Real.sin 1 + (Real.pi / 2)) :=
sorry

end NUMINAMATH_GPT_range_of_y_l1319_131915


namespace NUMINAMATH_GPT_route_Y_saves_2_minutes_l1319_131952

noncomputable def distance_X : ℝ := 8
noncomputable def speed_X : ℝ := 40

noncomputable def distance_Y1 : ℝ := 5
noncomputable def speed_Y1 : ℝ := 50
noncomputable def distance_Y2 : ℝ := 1
noncomputable def speed_Y2 : ℝ := 20
noncomputable def distance_Y3 : ℝ := 1
noncomputable def speed_Y3 : ℝ := 60

noncomputable def t_X : ℝ := (distance_X / speed_X) * 60
noncomputable def t_Y1 : ℝ := (distance_Y1 / speed_Y1) * 60
noncomputable def t_Y2 : ℝ := (distance_Y2 / speed_Y2) * 60
noncomputable def t_Y3 : ℝ := (distance_Y3 / speed_Y3) * 60
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3

noncomputable def time_saved : ℝ := t_X - t_Y

theorem route_Y_saves_2_minutes :
  time_saved = 2 := by
  sorry

end NUMINAMATH_GPT_route_Y_saves_2_minutes_l1319_131952


namespace NUMINAMATH_GPT_intersection_complement_l1319_131994

open Set

def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_complement :
  A ∩ (univ \ B) = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1319_131994


namespace NUMINAMATH_GPT_linear_eq_a_l1319_131965

theorem linear_eq_a (a : ℝ) (x y : ℝ) (h1 : (a + 1) ≠ 0) (h2 : |a| = 1) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_a_l1319_131965


namespace NUMINAMATH_GPT_max_area_of_rectangular_garden_l1319_131976

noncomputable def max_rectangle_area (x y : ℝ) (h1 : 2 * (x + y) = 36) (h2 : x > 0) (h3 : y > 0) : ℝ :=
  x * y

theorem max_area_of_rectangular_garden
  (x y : ℝ)
  (h1 : 2 * (x + y) = 36)
  (h2 : x > 0)
  (h3 : y > 0) :
  max_rectangle_area x y h1 h2 h3 = 81 :=
sorry

end NUMINAMATH_GPT_max_area_of_rectangular_garden_l1319_131976


namespace NUMINAMATH_GPT_find_divisor_l1319_131904

-- Define the given and calculated values in the conditions
def initial_value : ℕ := 165826
def subtracted_value : ℕ := 2
def resulting_value : ℕ := initial_value - subtracted_value

-- Define the goal: to find the smallest divisor of resulting_value other than 1
theorem find_divisor (d : ℕ) (h1 : initial_value - subtracted_value = resulting_value)
  (h2 : resulting_value % d = 0) (h3 : d > 1) : d = 2 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l1319_131904


namespace NUMINAMATH_GPT_sum_modulo_nine_l1319_131928

theorem sum_modulo_nine :
  (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_modulo_nine_l1319_131928


namespace NUMINAMATH_GPT_cube_sqrt_three_eq_three_sqrt_three_l1319_131978

theorem cube_sqrt_three_eq_three_sqrt_three : (Real.sqrt 3) ^ 3 = 3 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_cube_sqrt_three_eq_three_sqrt_three_l1319_131978


namespace NUMINAMATH_GPT_arithmetic_mean_x_is_16_point_4_l1319_131980

theorem arithmetic_mean_x_is_16_point_4 {x : ℝ}
  (h : (x + 10 + 17 + 2 * x + 15 + 2 * x + 6) / 5 = 26):
  x = 16.4 := 
sorry

end NUMINAMATH_GPT_arithmetic_mean_x_is_16_point_4_l1319_131980


namespace NUMINAMATH_GPT_product_discount_l1319_131968

theorem product_discount (P : ℝ) (h₁ : P > 0) :
  let price_after_first_discount := 0.7 * P
  let price_after_second_discount := 0.8 * price_after_first_discount
  let total_reduction := P - price_after_second_discount
  let percent_reduction := (total_reduction / P) * 100
  percent_reduction = 44 :=
by
  sorry

end NUMINAMATH_GPT_product_discount_l1319_131968


namespace NUMINAMATH_GPT_Isabel_total_problems_l1319_131985

theorem Isabel_total_problems :
  let math_pages := 2
  let reading_pages := 4
  let science_pages := 3
  let history_pages := 1
  let problems_per_math_page := 5
  let problems_per_reading_page := 5
  let problems_per_science_page := 7
  let problems_per_history_page := 10
  let total_math_problems := math_pages * problems_per_math_page
  let total_reading_problems := reading_pages * problems_per_reading_page
  let total_science_problems := science_pages * problems_per_science_page
  let total_history_problems := history_pages * problems_per_history_page
  let total_problems := total_math_problems + total_reading_problems + total_science_problems + total_history_problems
  total_problems = 61 := by
  sorry

end NUMINAMATH_GPT_Isabel_total_problems_l1319_131985


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1319_131990

universe u

variables {Point : Type u} 
variables (Plane : Type u) (Line : Type u)
variables (α β : Plane) (l : Line)
variables (P Q : Point)
variables (is_perpendicular : Plane → Plane → Prop)
variables (is_on_plane : Point → Plane → Prop)
variables (is_on_line : Point → Line → Prop)
variables (PQ_perpendicular_to_l : Prop) 
variables (PQ_perpendicular_to_β : Prop)
variables (line_in_plane : Line → Plane → Prop)

-- Given conditions
axiom plane_perpendicular : is_perpendicular α β
axiom plane_intersection : ∀ (α β : Plane), is_perpendicular α β → ∃ l : Line, line_in_plane l β
axiom point_on_plane_alpha : is_on_plane P α
axiom point_on_line : is_on_line Q l

-- Problem statement
theorem necessary_and_sufficient_condition :
  (PQ_perpendicular_to_l ↔ PQ_perpendicular_to_β) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1319_131990


namespace NUMINAMATH_GPT_int_as_sum_of_squares_l1319_131954

theorem int_as_sum_of_squares (n : ℤ) : ∃ a b c : ℤ, n = a^2 + b^2 - c^2 :=
sorry

end NUMINAMATH_GPT_int_as_sum_of_squares_l1319_131954


namespace NUMINAMATH_GPT_circle_radius_l1319_131981

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y = 0) : ∃ r : ℝ, r = Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1319_131981


namespace NUMINAMATH_GPT_min_value_frac_sum_l1319_131962

theorem min_value_frac_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1): 
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l1319_131962


namespace NUMINAMATH_GPT_problem_statement_l1319_131939

def A : Prop := (∀ (x : ℝ), x^2 - 3*x + 2 = 0 → x = 2)
def B : Prop := (∃ (x : ℝ), x^2 - x + 1 < 0)
def C : Prop := (¬(∀ (x : ℝ), x > 2 → x^2 - 3*x + 2 > 0))

theorem problem_statement :
  ¬ (A ∧ ∀ (x : ℝ), (B → (x^2 - x + 1) ≥ 0) ∧ (¬(A) ∧ C)) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1319_131939


namespace NUMINAMATH_GPT_remaining_digits_count_l1319_131995

theorem remaining_digits_count 
  (avg9 : ℝ) (avg4 : ℝ) (avgRemaining : ℝ) (h1 : avg9 = 18) (h2 : avg4 = 8) (h3 : avgRemaining = 26) :
  let S := 9 * avg9
  let S4 := 4 * avg4
  let S_remaining := S - S4
  let N := S_remaining / avgRemaining
  N = 5 := 
by
  sorry

end NUMINAMATH_GPT_remaining_digits_count_l1319_131995


namespace NUMINAMATH_GPT_flagpole_break_height_l1319_131956

theorem flagpole_break_height (total_height break_point distance_from_base : ℝ) 
(h_total : total_height = 6) 
(h_distance : distance_from_base = 2) 
(h_equation : (distance_from_base^2 + (total_height - break_point)^2) = break_point^2) :
  break_point = 3 := 
sorry

end NUMINAMATH_GPT_flagpole_break_height_l1319_131956


namespace NUMINAMATH_GPT_remainder_when_divided_by_22_l1319_131907

theorem remainder_when_divided_by_22 (y : ℤ) (k : ℤ) (h : y = 264 * k + 42) : y % 22 = 20 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_22_l1319_131907


namespace NUMINAMATH_GPT_number_of_dogs_l1319_131930

theorem number_of_dogs (total_animals cats : ℕ) (probability : ℚ) (h1 : total_animals = 7) (h2 : cats = 2) (h3 : probability = 2 / 7) :
  total_animals - cats = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_l1319_131930


namespace NUMINAMATH_GPT_domain_cannot_be_0_to_3_l1319_131914

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Define the range of the function f
def range_f : Set ℝ := Set.Icc 1 2

-- Statement that the domain [0, 3] cannot be the domain of f given the range
theorem domain_cannot_be_0_to_3 :
  ∀ (f : ℝ → ℝ) (range_f : Set ℝ),
    (∀ x, 1 ≤ f x ∧ f x ≤ 2) →
    ¬ ∃ dom : Set ℝ, dom = Set.Icc 0 3 ∧ 
      (∀ x ∈ dom, f x ∈ range_f) :=
by
  sorry

end NUMINAMATH_GPT_domain_cannot_be_0_to_3_l1319_131914


namespace NUMINAMATH_GPT_increase_in_difference_between_strawberries_and_blueberries_l1319_131997

theorem increase_in_difference_between_strawberries_and_blueberries :
  ∀ (B S : ℕ), B = 32 → S = B + 12 → (S - B) = 12 :=
by
  intros B S hB hS
  sorry

end NUMINAMATH_GPT_increase_in_difference_between_strawberries_and_blueberries_l1319_131997


namespace NUMINAMATH_GPT_find_triangle_height_l1319_131913

-- Define the problem conditions
def Rectangle.perimeter (l : ℕ) (w : ℕ) : ℕ := 2 * l + 2 * w
def Rectangle.area (l : ℕ) (w : ℕ) : ℕ := l * w
def Triangle.area (b : ℕ) (h : ℕ) : ℕ := (b * h) / 2

-- Conditions
namespace Conditions
  -- Perimeter of the rectangle is 60 cm
  def rect_perimeter (l w : ℕ) : Prop := Rectangle.perimeter l w = 60
  -- Base of the right triangle is 15 cm
  def tri_base : ℕ := 15
  -- Areas of the rectangle and the triangle are equal
  def equal_areas (l w h : ℕ) : Prop := Rectangle.area l w = Triangle.area tri_base h
end Conditions

-- Proof problem: Given these conditions, prove h = 30
theorem find_triangle_height (l w h : ℕ) 
  (h1 : Conditions.rect_perimeter l w)
  (h2 : Conditions.equal_areas l w h) : h = 30 :=
  sorry

end NUMINAMATH_GPT_find_triangle_height_l1319_131913


namespace NUMINAMATH_GPT_Lulu_blueberry_pies_baked_l1319_131912

-- Definitions of conditions
def Lola_mini_cupcakes := 13
def Lola_pop_tarts := 10
def Lola_blueberry_pies := 8
def Lola_total_pastries := Lola_mini_cupcakes + Lola_pop_tarts + Lola_blueberry_pies
def Lulu_mini_cupcakes := 16
def Lulu_pop_tarts := 12
def total_pastries := 73

-- Prove that Lulu baked 14 blueberry pies
theorem Lulu_blueberry_pies_baked : 
  ∃ (Lulu_blueberry_pies : Nat), 
    Lola_total_pastries + Lulu_mini_cupcakes + Lulu_pop_tarts + Lulu_blueberry_pies = total_pastries ∧ 
    Lulu_blueberry_pies = 14 := by
  sorry

end NUMINAMATH_GPT_Lulu_blueberry_pies_baked_l1319_131912


namespace NUMINAMATH_GPT_fg_of_neg3_eq_3_l1319_131971

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_of_neg3_eq_3 : f (g (-3)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_fg_of_neg3_eq_3_l1319_131971


namespace NUMINAMATH_GPT_kaleb_saved_initial_amount_l1319_131927

theorem kaleb_saved_initial_amount (allowance toys toy_price : ℕ) (total_savings : ℕ)
  (h1 : allowance = 15)
  (h2 : toys = 6)
  (h3 : toy_price = 6)
  (h4 : total_savings = toys * toy_price - allowance) :
  total_savings = 21 :=
  sorry

end NUMINAMATH_GPT_kaleb_saved_initial_amount_l1319_131927


namespace NUMINAMATH_GPT_algebraic_simplification_l1319_131901

variables (a b : ℝ)

theorem algebraic_simplification (h : a > b ∧ b > 0) : 
  ((a + b) / ((Real.sqrt a - Real.sqrt b)^2)) * 
  (((3 * a * b - b * Real.sqrt (a * b) + a * Real.sqrt (a * b) - 3 * b^2) / 
    (1/2 * Real.sqrt (1/4 * ((a / b + b / a)^2) - 1)) + 
   (4 * a * b * Real.sqrt a + 9 * a * b * Real.sqrt b - 9 * b^2 * Real.sqrt a) / 
   (3/2 * Real.sqrt b - 2 * Real.sqrt a))) 
  = -2 * b * (a + 3 * Real.sqrt (a * b)) :=
sorry

end NUMINAMATH_GPT_algebraic_simplification_l1319_131901


namespace NUMINAMATH_GPT_analyze_monotonicity_and_find_a_range_l1319_131958

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem analyze_monotonicity_and_find_a_range
  (a : ℝ)
  (h : ∀ x : ℝ, f x a + f_prime x a = 2 - a * x^2) :
  (∀ x : ℝ, a ≤ 0 → f_prime x a > 0) ∧
  (a > 0 → (∀ x : ℝ, (x < Real.log (2 * a) → f_prime x a < 0) ∧ (x > Real.log (2 * a) → f_prime x a > 0))) ∧
  (1 < a ∧ a < Real.exp 1 - 1) :=
sorry

end NUMINAMATH_GPT_analyze_monotonicity_and_find_a_range_l1319_131958


namespace NUMINAMATH_GPT_focus_of_parabola_l1319_131905

theorem focus_of_parabola (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) ∈ {p : ℝ × ℝ | ∃ x y, y = 4 * x^2 ∧ p = (0, 1 / (4 * (1 / y)))} :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l1319_131905


namespace NUMINAMATH_GPT_frequency_of_fourth_group_l1319_131906

theorem frequency_of_fourth_group (f₁ f₂ f₃ f₄ f₅ f₆ : ℝ) (h1 : f₁ + f₂ + f₃ = 0.65) (h2 : f₅ + f₆ = 0.32) (h3 : f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = 1) :
  f₄ = 0.03 :=
by 
  sorry

end NUMINAMATH_GPT_frequency_of_fourth_group_l1319_131906


namespace NUMINAMATH_GPT_max_abs_value_l1319_131933

theorem max_abs_value (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : |x - 2 * y + 1| ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_max_abs_value_l1319_131933


namespace NUMINAMATH_GPT_range_of_m_l1319_131992

noncomputable def f (x : ℝ) := Real.log (x^2 + 1)

noncomputable def g (x m : ℝ) := (1 / 2)^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (0:ℝ) 3, ∃ x2 ∈ Set.Icc (1:ℝ) 2, f x1 ≤ g x2 m) ↔ m ≤ -1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1319_131992


namespace NUMINAMATH_GPT_value_of_expression_l1319_131953

theorem value_of_expression (x y : ℝ) (h1 : 3 * x + 2 * y = 7) (h2 : 2 * x + 3 * y = 8) :
  13 * x ^ 2 + 22 * x * y + 13 * y ^ 2 = 113 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1319_131953


namespace NUMINAMATH_GPT_product_B_sampling_l1319_131973

theorem product_B_sampling (a : ℕ) (h_seq : a > 0) :
  let A := a
  let B := 2 * a
  let C := 4 * a
  let total := A + B + C
  total = 7 * a →
  let total_drawn := 140
  B / total * total_drawn = 40 :=
by sorry

end NUMINAMATH_GPT_product_B_sampling_l1319_131973


namespace NUMINAMATH_GPT_sugar_left_correct_l1319_131921

-- Define the total amount of sugar bought by Pamela
def total_sugar : ℝ := 9.8

-- Define the amount of sugar spilled by Pamela
def spilled_sugar : ℝ := 5.2

-- Define the amount of sugar left after spilling
def sugar_left : ℝ := total_sugar - spilled_sugar

-- State that the amount of sugar left should be equivalent to the correct answer
theorem sugar_left_correct : sugar_left = 4.6 :=
by
  sorry

end NUMINAMATH_GPT_sugar_left_correct_l1319_131921


namespace NUMINAMATH_GPT_positive_m_of_quadratic_has_one_real_root_l1319_131932

theorem positive_m_of_quadratic_has_one_real_root : 
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, x^2 + 6 * m * x + m = 0 → x = -3 * m) :=
by
  sorry

end NUMINAMATH_GPT_positive_m_of_quadratic_has_one_real_root_l1319_131932


namespace NUMINAMATH_GPT_minValue_expression_l1319_131944

theorem minValue_expression (x y : ℝ) (h : x + 2 * y = 4) : ∃ (v : ℝ), v = 2^x + 4^y ∧ ∀ (a b : ℝ), a + 2 * b = 4 → 2^a + 4^b ≥ v :=
by 
  sorry

end NUMINAMATH_GPT_minValue_expression_l1319_131944


namespace NUMINAMATH_GPT_alyssa_gave_away_puppies_l1319_131926

def start_puppies : ℕ := 12
def remaining_puppies : ℕ := 5

theorem alyssa_gave_away_puppies : 
  start_puppies - remaining_puppies = 7 := 
by
  sorry

end NUMINAMATH_GPT_alyssa_gave_away_puppies_l1319_131926


namespace NUMINAMATH_GPT_solve_equation_parabola_equation_l1319_131988

-- Part 1: Equation Solutions
theorem solve_equation {x : ℝ} :
  (x - 9) ^ 2 = 2 * (x - 9) ↔ x = 9 ∨ x = 11 := by
  sorry

-- Part 2: Expression of Parabola
theorem parabola_equation (a h k : ℝ) (vertex : (ℝ × ℝ)) (point: (ℝ × ℝ)) :
  vertex = (-3, 2) → point = (-1, -2) →
  a * (point.1 - h) ^ 2 + k = point.2 →
  (a = -1) → (h = -3) → (k = 2) →
  - x ^ 2 - 6 * x - 7 = a * (x + 3) ^ 2 + 2 := by
  sorry

end NUMINAMATH_GPT_solve_equation_parabola_equation_l1319_131988


namespace NUMINAMATH_GPT_power_sum_positive_l1319_131969

theorem power_sum_positive 
    (a b c : ℝ) 
    (h1 : a * b * c > 0)
    (h2 : a + b + c > 0)
    (n : ℕ):
    a ^ n + b ^ n + c ^ n > 0 :=
by
  sorry

end NUMINAMATH_GPT_power_sum_positive_l1319_131969


namespace NUMINAMATH_GPT_binomial_coefficient_is_252_l1319_131966

theorem binomial_coefficient_is_252 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_is_252_l1319_131966


namespace NUMINAMATH_GPT_percentage_increase_l1319_131923

theorem percentage_increase (original_price new_price : ℝ) (h₀ : original_price = 300) (h₁ : new_price = 420) :
  ((new_price - original_price) / original_price) * 100 = 40 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_percentage_increase_l1319_131923


namespace NUMINAMATH_GPT_reciprocal_neg_one_over_2023_eq_neg_2023_l1319_131998

theorem reciprocal_neg_one_over_2023_eq_neg_2023 : (1 / (-1 / (2023 : ℝ))) = -2023 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_one_over_2023_eq_neg_2023_l1319_131998


namespace NUMINAMATH_GPT_problem_solution_l1319_131996

variable (y Q : ℝ)

theorem problem_solution
  (h : 4 * (5 * y + 3 * Real.pi) = Q) :
  8 * (10 * y + 6 * Real.pi + 2 * Real.sqrt 3) = 4 * Q + 16 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1319_131996


namespace NUMINAMATH_GPT_sum_of_remainders_l1319_131940

theorem sum_of_remainders (a b c : ℕ) 
  (ha : a % 15 = 12) 
  (hb : b % 15 = 13) 
  (hc : c % 15 = 14) : 
  (a + b + c) % 15 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1319_131940


namespace NUMINAMATH_GPT_john_uber_profit_l1319_131919

theorem john_uber_profit
  (P0 : ℝ) (T : ℝ) (P : ℝ)
  (hP0 : P0 = 18000)
  (hT : T = 6000)
  (hP : P = 18000) :
  P + (P0 - T) = 30000 :=
by
  sorry

end NUMINAMATH_GPT_john_uber_profit_l1319_131919


namespace NUMINAMATH_GPT_andrew_subway_time_l1319_131947

variable (S : ℝ) -- Let \( S \) be the time Andrew spends on the subway in hours

variable (total_time : ℝ)
variable (bike_time : ℝ)
variable (train_time : ℝ)

noncomputable def travel_conditions := 
  total_time = S + 2 * S + bike_time ∧ 
  total_time = 38 ∧ 
  bike_time = 8

theorem andrew_subway_time
  (S : ℝ)
  (total_time : ℝ)
  (bike_time : ℝ)
  (train_time : ℝ)
  (h : travel_conditions S total_time bike_time) : 
  S = 10 := 
sorry

end NUMINAMATH_GPT_andrew_subway_time_l1319_131947


namespace NUMINAMATH_GPT_find_sum_of_distinct_real_numbers_l1319_131942

noncomputable def determinant_3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem find_sum_of_distinct_real_numbers (x y : ℝ) (hxy : x ≠ y) 
    (h : determinant_3x3 1 6 15 3 x y 3 y x = 0) : x + y = 63 := 
by
  sorry

end NUMINAMATH_GPT_find_sum_of_distinct_real_numbers_l1319_131942


namespace NUMINAMATH_GPT_emily_needs_375_nickels_for_book_l1319_131924

theorem emily_needs_375_nickels_for_book
  (n : ℕ)
  (book_cost : ℝ)
  (five_dollars : ℝ)
  (one_dollars : ℝ)
  (quarters : ℝ)
  (nickel_value : ℝ)
  (total_money : ℝ)
  (h1 : book_cost = 46.25)
  (h2 : five_dollars = 4 * 5)
  (h3 : one_dollars = 5 * 1)
  (h4 : quarters = 10 * 0.25)
  (h5 : nickel_value = n * 0.05)
  (h6 : total_money = five_dollars + one_dollars + quarters + nickel_value) 
  (h7 : total_money ≥ book_cost) :
  n ≥ 375 :=
by 
  sorry

end NUMINAMATH_GPT_emily_needs_375_nickels_for_book_l1319_131924


namespace NUMINAMATH_GPT_prove_y_l1319_131984

-- Define the conditions
variables (x y : ℤ) -- x and y are integers

-- State the problem conditions
def conditions := (x + y = 270) ∧ (x - y = 200)

-- Define the theorem to prove that y = 35 given the conditions
theorem prove_y : conditions x y → y = 35 :=
by
  sorry

end NUMINAMATH_GPT_prove_y_l1319_131984


namespace NUMINAMATH_GPT_frustum_lateral_surface_area_l1319_131957

theorem frustum_lateral_surface_area:
  ∀ (R r h : ℝ), R = 7 → r = 4 → h = 6 → (∃ L, L = 33 * Real.pi * Real.sqrt 5) := by
  sorry

end NUMINAMATH_GPT_frustum_lateral_surface_area_l1319_131957


namespace NUMINAMATH_GPT_total_coffee_cost_l1319_131929

def vacation_days : ℕ := 40
def daily_coffee : ℕ := 3
def pods_per_box : ℕ := 30
def box_cost : ℕ := 8

theorem total_coffee_cost : vacation_days * daily_coffee / pods_per_box * box_cost = 32 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_coffee_cost_l1319_131929


namespace NUMINAMATH_GPT_range_of_y_eq_x_squared_l1319_131987

noncomputable def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem range_of_y_eq_x_squared :
  M = { y : ℝ | ∃ x : ℝ, y = x^2 } := by
  sorry

end NUMINAMATH_GPT_range_of_y_eq_x_squared_l1319_131987


namespace NUMINAMATH_GPT_camden_total_legs_l1319_131920

theorem camden_total_legs 
  (num_justin_dogs : ℕ := 14)
  (num_rico_dogs := num_justin_dogs + 10)
  (num_camden_dogs := 3 * num_rico_dogs / 4)
  (camden_3_leg_dogs : ℕ := 5)
  (camden_4_leg_dogs : ℕ := 7)
  (camden_2_leg_dogs : ℕ := 2) : 
  3 * camden_3_leg_dogs + 4 * camden_4_leg_dogs + 2 * camden_2_leg_dogs = 47 :=
by sorry

end NUMINAMATH_GPT_camden_total_legs_l1319_131920


namespace NUMINAMATH_GPT_repeating_decimal_is_fraction_l1319_131977

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_is_fraction_l1319_131977


namespace NUMINAMATH_GPT_sum_of_octal_numbers_l1319_131979

theorem sum_of_octal_numbers :
  let a := 0o1275
  let b := 0o164
  let sum := 0o1503
  a + b = sum :=
by
  -- Proof is omitted here with sorry
  sorry

end NUMINAMATH_GPT_sum_of_octal_numbers_l1319_131979


namespace NUMINAMATH_GPT_function_tangent_and_max_k_l1319_131948

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2 * x - 1

theorem function_tangent_and_max_k 
  (x : ℝ) (h1 : 0 < x) 
  (h2 : 3 * x - y - 2 = 0) : 
  (∀ k : ℤ, (∀ x : ℝ, 1 < x → k < (f x) / (x - 1)) → k ≤ 4) := 
sorry

end NUMINAMATH_GPT_function_tangent_and_max_k_l1319_131948


namespace NUMINAMATH_GPT_expression_evaluation_l1319_131950

variable (x y : ℝ)

theorem expression_evaluation (h1 : x = 2 * y) (h2 : y ≠ 0) : 
  (x + 2 * y) - (2 * x + y) = -y := 
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1319_131950


namespace NUMINAMATH_GPT_all_boxcars_combined_capacity_l1319_131918

theorem all_boxcars_combined_capacity :
  let black_capacity := 4000
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  let green_capacity := 1.5 * black_capacity
  let yellow_capacity := green_capacity + 2000
  let total_red := 3 * red_capacity
  let total_blue := 4 * blue_capacity
  let total_black := 7 * black_capacity
  let total_green := 2 * green_capacity
  let total_yellow := 5 * yellow_capacity
  total_red + total_blue + total_black + total_green + total_yellow = 184000 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_all_boxcars_combined_capacity_l1319_131918


namespace NUMINAMATH_GPT_krista_egg_sales_l1319_131946

-- Define the conditions
def hens : ℕ := 10
def eggs_per_hen_per_week : ℕ := 12
def price_per_dozen : ℕ := 3
def weeks : ℕ := 4

-- Define the total money made as the value we want to prove
def total_money_made : ℕ := 120

-- State the theorem
theorem krista_egg_sales : 
  (hens * eggs_per_hen_per_week * weeks / 12) * price_per_dozen = total_money_made :=
by
  sorry

end NUMINAMATH_GPT_krista_egg_sales_l1319_131946


namespace NUMINAMATH_GPT_negation_proposition_l1319_131986

theorem negation_proposition :
  (¬ (∀ x : ℝ, abs x + x^2 ≥ 0)) ↔ (∃ x₀ : ℝ, abs x₀ + x₀^2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1319_131986


namespace NUMINAMATH_GPT_lorelei_vase_rose_count_l1319_131909

theorem lorelei_vase_rose_count :
  let r := 12
  let p := 18
  let y := 20
  let o := 8
  let picked_r := 0.5 * r
  let picked_p := 0.5 * p
  let picked_y := 0.25 * y
  let picked_o := 0.25 * o
  picked_r + picked_p + picked_y + picked_o = 22 := 
by
  sorry

end NUMINAMATH_GPT_lorelei_vase_rose_count_l1319_131909


namespace NUMINAMATH_GPT_total_brownies_correct_l1319_131963

noncomputable def initial_brownies : ℕ := 2 * 12
noncomputable def brownies_after_father : ℕ := initial_brownies - 8
noncomputable def brownies_after_mooney : ℕ := brownies_after_father - 4
noncomputable def additional_brownies : ℕ := 2 * 12
noncomputable def total_brownies : ℕ := brownies_after_mooney + additional_brownies

theorem total_brownies_correct : total_brownies = 36 := by
  sorry

end NUMINAMATH_GPT_total_brownies_correct_l1319_131963


namespace NUMINAMATH_GPT_binary_10101000_is_1133_base_5_l1319_131955

def binary_to_decimal (b : Nat) : Nat :=
  128 * (b / 128 % 2) + 64 * (b / 64 % 2) + 32 * (b / 32 % 2) + 16 * (b / 16 % 2) + 8 * (b / 8 % 2) + 4 * (b / 4 % 2) + 2 * (b / 2 % 2) + (b % 2)

def decimal_to_base_5 (d : Nat) : List Nat :=
  if d = 0 then [] else (d % 5) :: decimal_to_base_5 (d / 5)

def binary_to_base_5 (b : Nat) : List Nat :=
  decimal_to_base_5 (binary_to_decimal b)

theorem binary_10101000_is_1133_base_5 :
  binary_to_base_5 168 = [1, 1, 3, 3] := 
by 
  sorry

end NUMINAMATH_GPT_binary_10101000_is_1133_base_5_l1319_131955


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l1319_131903

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a S : ℕ → ℚ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def condition_1 (a : ℕ → ℚ) : Prop :=
  is_arithmetic_sequence a

def condition_2 (a : ℕ → ℚ) : Prop :=
  (a 5) / (a 3) = 5 / 9

-- Proof statement
theorem arithmetic_sequence_sum_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : condition_1 a) (h2 : condition_2 a) (h3 : sum_of_first_n_terms a S) : 
  (S 9) / (S 5) = 1 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l1319_131903


namespace NUMINAMATH_GPT_no_solution_eq_l1319_131902

theorem no_solution_eq (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → ((3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) → (m = -1)) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_eq_l1319_131902


namespace NUMINAMATH_GPT_modulus_of_complex_division_l1319_131938

noncomputable def complexDivisionModulus : ℂ := Complex.normSq (2 * Complex.I / (Complex.I - 1))

theorem modulus_of_complex_division : complexDivisionModulus = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_modulus_of_complex_division_l1319_131938


namespace NUMINAMATH_GPT_kitten_food_consumption_l1319_131917

-- Definitions of the given conditions
def k : ℕ := 4  -- Number of kittens
def ac : ℕ := 3  -- Number of adult cats
def f : ℕ := 7  -- Initial cans of food
def af : ℕ := 35  -- Additional cans of food needed
def days : ℕ := 7  -- Total number of days

-- Definition of the food consumption per adult cat per day
def food_per_adult_cat_per_day : ℕ := 1

-- Definition of the correct answer: food per kitten per day
def food_per_kitten_per_day : ℚ := 0.75

-- Proof statement
theorem kitten_food_consumption (k : ℕ) (ac : ℕ) (f : ℕ) (af : ℕ) (days : ℕ) (food_per_adult_cat_per_day : ℕ) :
  (ac * food_per_adult_cat_per_day * days + k * food_per_kitten_per_day * days = f + af) → 
  food_per_kitten_per_day = 0.75 :=
sorry

end NUMINAMATH_GPT_kitten_food_consumption_l1319_131917


namespace NUMINAMATH_GPT_cost_of_four_pencils_and_three_pens_l1319_131982

variable {p q : ℝ}

theorem cost_of_four_pencils_and_three_pens (h1 : 3 * p + 2 * q = 4.30) (h2 : 2 * p + 3 * q = 4.05) : 4 * p + 3 * q = 5.97 := by
  sorry

end NUMINAMATH_GPT_cost_of_four_pencils_and_three_pens_l1319_131982


namespace NUMINAMATH_GPT_find_g_inv_84_l1319_131941

def g (x : ℝ) : ℝ := 3 * x ^ 3 + 3

theorem find_g_inv_84 (x : ℝ) (h : g x = 84) : x = 3 :=
by 
  unfold g at h
  -- Begin proof steps here, but we will use sorry to denote placeholder 

  sorry

end NUMINAMATH_GPT_find_g_inv_84_l1319_131941


namespace NUMINAMATH_GPT_solution_set_inequality_l1319_131937

theorem solution_set_inequality (x : ℝ) : 
  ((x-2) * (3-x) > 0) ↔ (2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l1319_131937


namespace NUMINAMATH_GPT_solve_k_l1319_131900

theorem solve_k (x y k : ℝ) (h1 : x + 2 * y = k - 1) (h2 : 2 * x + y = 5 * k + 4) (h3 : x + y = 5) :
  k = 2 :=
sorry

end NUMINAMATH_GPT_solve_k_l1319_131900


namespace NUMINAMATH_GPT_expression_evaluation_l1319_131910

theorem expression_evaluation :
  2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := sorry

end NUMINAMATH_GPT_expression_evaluation_l1319_131910


namespace NUMINAMATH_GPT_width_of_rectangular_field_l1319_131961

theorem width_of_rectangular_field
  (L W : ℝ)
  (h1 : L = (7/5) * W)
  (h2 : 2 * L + 2 * W = 384) :
  W = 80 :=
by
  sorry

end NUMINAMATH_GPT_width_of_rectangular_field_l1319_131961


namespace NUMINAMATH_GPT_simplify_fraction_l1319_131959

theorem simplify_fraction:
  (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1319_131959


namespace NUMINAMATH_GPT_one_integral_root_exists_l1319_131972

theorem one_integral_root_exists :
    ∃! x : ℤ, x - 8 / (x - 3) = 2 - 8 / (x - 3) :=
by
  sorry

end NUMINAMATH_GPT_one_integral_root_exists_l1319_131972


namespace NUMINAMATH_GPT_pupils_count_l1319_131974

-- Definitions based on given conditions
def number_of_girls : ℕ := 692
def girls_more_than_boys : ℕ := 458
def number_of_boys : ℕ := number_of_girls - girls_more_than_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

-- The statement that the total number of pupils is 926
theorem pupils_count : total_pupils = 926 := by
  sorry

end NUMINAMATH_GPT_pupils_count_l1319_131974


namespace NUMINAMATH_GPT_units_digit_S7890_l1319_131916

noncomputable def c : ℝ := 4 + 3 * Real.sqrt 2
noncomputable def d : ℝ := 4 - 3 * Real.sqrt 2
noncomputable def S (n : ℕ) : ℝ := (1/2:ℝ) * (c^n + d^n)

theorem units_digit_S7890 : (S 7890) % 10 = 8 :=
sorry

end NUMINAMATH_GPT_units_digit_S7890_l1319_131916


namespace NUMINAMATH_GPT_find_units_digit_l1319_131922

def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem find_units_digit (n : ℕ) :
  is_three_digit n →
  (is_perfect_square n ∨ is_even n ∨ is_divisible_by_11 n ∨ digit_sum n = 12) ∧
  (¬is_perfect_square n ∨ ¬is_even n ∨ ¬is_divisible_by_11 n ∨ ¬(digit_sum n = 12)) →
  (n % 10 = 4) :=
sorry

end NUMINAMATH_GPT_find_units_digit_l1319_131922


namespace NUMINAMATH_GPT_bullying_instances_l1319_131911

-- Let's denote the total number of suspension days due to bullying and serious incidents.
def total_suspension_days : ℕ := (3 * (10 + 10)) + 14

-- Each instance of bullying results in a 3-day suspension.
def days_per_instance : ℕ := 3

-- The number of instances of bullying given the total suspension days.
def instances_of_bullying := total_suspension_days / days_per_instance

-- We must prove that Kris is responsible for 24 instances of bullying.
theorem bullying_instances : instances_of_bullying = 24 := by
  sorry

end NUMINAMATH_GPT_bullying_instances_l1319_131911


namespace NUMINAMATH_GPT_shadedQuadrilateralArea_is_13_l1319_131991

noncomputable def calculateShadedQuadrilateralArea : ℝ :=
  let s1 := 2
  let s2 := 4
  let s3 := 6
  let s4 := 8
  let bases := s1 + s2
  let height_small := bases * (10 / 20)
  let height_large := 10
  let alt := s4 - s3
  let area := (1 / 2) * (height_small + height_large) * alt
  13

theorem shadedQuadrilateralArea_is_13 :
  calculateShadedQuadrilateralArea = 13 := by
sorry

end NUMINAMATH_GPT_shadedQuadrilateralArea_is_13_l1319_131991


namespace NUMINAMATH_GPT_anna_chocolates_l1319_131931

theorem anna_chocolates : ∃ (n : ℕ), (5 * 2^(n-1) > 200) ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_anna_chocolates_l1319_131931


namespace NUMINAMATH_GPT_probability_heart_then_club_l1319_131964

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end NUMINAMATH_GPT_probability_heart_then_club_l1319_131964


namespace NUMINAMATH_GPT_f_17_l1319_131960

def f : ℕ → ℤ := sorry

axiom f_prop1 : f 1 = 0
axiom f_prop2 : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) = f m + f n + 4 * (9 * m * n - 1)

theorem f_17 : f 17 = 1052 := by
  sorry

end NUMINAMATH_GPT_f_17_l1319_131960


namespace NUMINAMATH_GPT_walking_time_l1319_131925

-- Define the conditions as Lean definitions
def minutes_in_hour : Nat := 60

def work_hours : Nat := 6
def work_minutes := work_hours * minutes_in_hour
def sitting_interval : Nat := 90
def walking_time_per_interval : Nat := 10

-- State the main theorem
theorem walking_time (h1 : 10 * 90 = 600) (h2 : 10 * (work_hours * 60) / 90 = 40) : 
  work_minutes / sitting_interval * walking_time_per_interval = 40 :=
  sorry

end NUMINAMATH_GPT_walking_time_l1319_131925


namespace NUMINAMATH_GPT_three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l1319_131989

theorem three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693 :
  ∃ (n : ℕ), n = 693 ∧ 
    (100 * 6 + 10 * (n / 10 % 10) + 3) = n ∧
    (n % 10 = 3) ∧
    (n / 100 = 6) ∧
    n % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_with_units_3_and_hundreds_6_divisible_by_11_is_693_l1319_131989


namespace NUMINAMATH_GPT_total_fleas_l1319_131975

-- Definitions based on conditions provided
def fleas_Gertrude : Nat := 10
def fleas_Olive : Nat := fleas_Gertrude / 2
def fleas_Maud : Nat := 5 * fleas_Olive

-- Prove the total number of fleas on all three chickens
theorem total_fleas :
  fleas_Gertrude + fleas_Olive + fleas_Maud = 40 :=
by sorry

end NUMINAMATH_GPT_total_fleas_l1319_131975


namespace NUMINAMATH_GPT_sum_of_cubes_inequality_l1319_131934

theorem sum_of_cubes_inequality (a b c : ℝ) (h1 : a >= -1) (h2 : b >= -1) (h3 : c >= -1) (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 <= 4 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_inequality_l1319_131934


namespace NUMINAMATH_GPT_parabola_opens_downwards_l1319_131967

theorem parabola_opens_downwards (m : ℝ) : (m + 3 < 0) → (m < -3) := 
by
  sorry

end NUMINAMATH_GPT_parabola_opens_downwards_l1319_131967


namespace NUMINAMATH_GPT_primes_dividing_sequence_l1319_131983

def a_n (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def is_prime (p : ℕ) := Nat.Prime p

theorem primes_dividing_sequence :
  {p : ℕ | is_prime p ∧ p ≤ 19 ∧ ∃ n ≥ 1, p ∣ a_n n} = {3, 7, 13, 17} :=
by
  sorry

end NUMINAMATH_GPT_primes_dividing_sequence_l1319_131983


namespace NUMINAMATH_GPT_packages_katie_can_make_l1319_131970

-- Definition of the given conditions
def number_of_cupcakes_baked := 18
def cupcakes_eaten_by_todd := 8
def cupcakes_per_package := 2

-- The main statement to prove
theorem packages_katie_can_make : 
  (number_of_cupcakes_baked - cupcakes_eaten_by_todd) / cupcakes_per_package = 5 :=
by
  -- Use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_packages_katie_can_make_l1319_131970
