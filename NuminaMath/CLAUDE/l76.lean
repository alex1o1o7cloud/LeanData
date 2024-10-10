import Mathlib

namespace value_of_a_l76_7614

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}
def A (a : ℝ) : Set ℝ := {2, |a + 1|}

theorem value_of_a (a : ℝ) : U a = A a ∪ {5} → a = -4 ∨ a = 2 := by
  sorry

end value_of_a_l76_7614


namespace first_degree_function_composition_l76_7682

theorem first_degree_function_composition (f : ℝ → ℝ) :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) →
  (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
by sorry

end first_degree_function_composition_l76_7682


namespace scientific_notation_of_0_02008_l76_7697

def original_number : ℝ := 0.02008

def scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_0_02008 :
  scientific_notation original_number 3 = (2.01, -2) :=
sorry

end scientific_notation_of_0_02008_l76_7697


namespace maci_school_supplies_cost_l76_7650

/-- Calculates the total cost of school supplies with discounts applied --/
def calculate_total_cost (blue_pen_count : ℕ) (red_pen_count : ℕ) (pencil_count : ℕ) (notebook_count : ℕ) 
  (blue_pen_price : ℚ) (pen_discount_threshold : ℕ) (pen_discount_rate : ℚ) 
  (notebook_discount_threshold : ℕ) (notebook_discount_rate : ℚ) : ℚ :=
  let red_pen_price := 2 * blue_pen_price
  let pencil_price := red_pen_price / 2
  let notebook_price := 10 * blue_pen_price
  
  let total_pen_cost := blue_pen_count * blue_pen_price + red_pen_count * red_pen_price
  let pencil_cost := pencil_count * pencil_price
  let notebook_cost := notebook_count * notebook_price
  
  let pen_discount := if blue_pen_count + red_pen_count > pen_discount_threshold 
                      then pen_discount_rate * total_pen_cost 
                      else 0
  let notebook_discount := if notebook_count > notebook_discount_threshold 
                           then notebook_discount_rate * notebook_cost 
                           else 0
  
  total_pen_cost + pencil_cost + notebook_cost - pen_discount - notebook_discount

/-- Theorem stating that the total cost of Maci's school supplies is $7.10 --/
theorem maci_school_supplies_cost :
  calculate_total_cost 10 15 5 3 (10/100) 12 (10/100) 4 (20/100) = 71/10 := by
  sorry

end maci_school_supplies_cost_l76_7650


namespace house_transaction_profit_l76_7617

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15

theorem house_transaction_profit : 
  let first_sale := initial_value * (1 + profit_percentage)
  let second_sale := first_sale * (1 - loss_percentage)
  first_sale - second_sale = 2700 := by sorry

end house_transaction_profit_l76_7617


namespace max_value_polynomial_l76_7664

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z : ℝ), x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ z) ∧
  (∀ (z : ℝ), x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ z → 6084/17 ≤ z) :=
sorry

end max_value_polynomial_l76_7664


namespace sqrt_inequality_equivalence_l76_7695

theorem sqrt_inequality_equivalence : 
  (Real.sqrt 2 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 7) ↔ 
  ((Real.sqrt 2 + Real.sqrt 7)^2 < (Real.sqrt 3 + Real.sqrt 6)^2) := by
sorry

end sqrt_inequality_equivalence_l76_7695


namespace identity_function_proof_l76_7684

theorem identity_function_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end identity_function_proof_l76_7684


namespace candy_distribution_l76_7613

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 22 → num_bags = 2 → candy_per_bag = total_candy / num_bags → candy_per_bag = 11 := by
  sorry

end candy_distribution_l76_7613


namespace signup_ways_eq_64_l76_7638

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of interest groups --/
def num_groups : ℕ := 4

/-- The number of ways students can sign up for interest groups --/
def num_ways : ℕ := num_groups ^ num_students

/-- Theorem stating that the number of ways to sign up is 64 --/
theorem signup_ways_eq_64 : num_ways = 64 := by
  sorry

end signup_ways_eq_64_l76_7638


namespace inequality_proof_l76_7681

/-- Given a function f: ℝ → ℝ with derivative f', such that ∀ x ∈ ℝ, f x > f' x,
    prove that 2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) -/
theorem inequality_proof (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : ∀ x : ℝ, HasDerivAt f (f' x) x)
    (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := by
  sorry

end inequality_proof_l76_7681


namespace rent_increase_percentage_l76_7660

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.10 * last_year_earnings
  let this_year_earnings := 1.15 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 345 :=
by sorry

end rent_increase_percentage_l76_7660


namespace sequence_general_term_l76_7663

theorem sequence_general_term (n : ℕ+) : 
  let a : ℕ+ → ℝ := fun i => Real.sqrt i
  a n = Real.sqrt n := by sorry

end sequence_general_term_l76_7663


namespace vasya_problem_impossible_l76_7631

theorem vasya_problem_impossible : ¬ ∃ (x₁ x₂ x₃ : ℕ), 
  x₁ + 3 * x₂ + 15 * x₃ = 100 ∧ 11 * x₁ + 8 * x₂ = 144 := by
  sorry

end vasya_problem_impossible_l76_7631


namespace smallest_a_value_l76_7639

/-- Given a polynomial x^3 - ax^2 + bx - 2550 with three positive integer roots,
    the smallest possible value of a is 62 -/
theorem smallest_a_value (a b : ℤ) (r s t : ℕ+) : 
  (∀ x, x^3 - a*x^2 + b*x - 2550 = (x - r.val)*(x - s.val)*(x - t.val)) →
  a = r.val + s.val + t.val →
  62 ≤ a :=
sorry

end smallest_a_value_l76_7639


namespace fraction_of_girls_l76_7632

theorem fraction_of_girls (total_students : ℕ) (boys : ℕ) (h1 : total_students = 160) (h2 : boys = 120) :
  (total_students - boys : ℚ) / total_students = 1 / 4 := by
  sorry

end fraction_of_girls_l76_7632


namespace sum_over_subsets_equals_power_of_two_l76_7641

def S : Finset Nat := Finset.range 1999

def f (X : Finset Nat) : Nat :=
  X.sum id

theorem sum_over_subsets_equals_power_of_two :
  (Finset.powerset S).sum (fun E => (f E : ℚ) / (f S : ℚ)) = (2 : ℚ) ^ 1998 :=
sorry

end sum_over_subsets_equals_power_of_two_l76_7641


namespace like_terms_exponent_l76_7616

theorem like_terms_exponent (a : ℝ) : (∃ x : ℝ, x ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ k * x^(2*a) = 5 * x^(a+3)) → a = 3 := by
  sorry

end like_terms_exponent_l76_7616


namespace lagrange_interpolation_identity_l76_7612

theorem lagrange_interpolation_identity 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  c^2 * ((x-a)*(x-b))/((c-a)*(c-b)) + 
  b^2 * ((x-a)*(x-c))/((b-a)*(b-c)) + 
  a^2 * ((x-b)*(x-c))/((a-b)*(a-c)) = x^2 := by
  sorry

end lagrange_interpolation_identity_l76_7612


namespace inequality_proof_l76_7699

theorem inequality_proof (x : ℝ) (h : 1 ≤ x ∧ x ≤ 5) : 2*x + 1/x + 1/(x+1) < 2 := by
  sorry

end inequality_proof_l76_7699


namespace mary_eggs_count_l76_7698

/-- Given that Mary starts with 27 eggs and finds 4 more eggs, prove that she ends up with 31 eggs in total. -/
theorem mary_eggs_count (initial_eggs found_eggs : ℕ) : 
  initial_eggs = 27 → found_eggs = 4 → initial_eggs + found_eggs = 31 := by
  sorry

end mary_eggs_count_l76_7698


namespace equal_percentage_price_l76_7679

/-- Represents the cost and various selling prices of an article -/
structure Article where
  cp : ℝ  -- Cost price
  sp_profit : ℝ  -- Selling price with 25% profit
  sp_loss : ℝ  -- Selling price with loss
  sp_equal : ℝ  -- Selling price where % profit = % loss

/-- Conditions for the article pricing problem -/
def article_conditions (a : Article) : Prop :=
  a.sp_profit = a.cp * 1.25 ∧  -- 25% profit condition
  a.sp_profit = 1625 ∧
  a.sp_loss = 1280 ∧
  a.sp_loss < a.cp  -- Ensures sp_loss results in a loss

/-- Theorem stating the selling price where percentage profit equals percentage loss -/
theorem equal_percentage_price (a : Article) 
  (h : article_conditions a) : a.sp_equal = 1320 := by
  sorry

#check equal_percentage_price

end equal_percentage_price_l76_7679


namespace skater_speed_l76_7642

/-- Given a skater who travels 80 kilometers in 8 hours, prove their speed is 10 kilometers per hour. -/
theorem skater_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 := by sorry

end skater_speed_l76_7642


namespace muirhead_inequality_l76_7658

open Real

/-- Muirhead's Inequality -/
theorem muirhead_inequality (a₁ a₂ a₃ b₁ b₂ b₃ x y z : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃ ∧ a₃ ≥ 0)
  (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃ ∧ b₃ ≥ 0)
  (hab : a₁ ≥ b₁ ∧ a₁ + a₂ ≥ b₁ + b₂ ∧ a₁ + a₂ + a₃ ≥ b₁ + b₂ + b₃)
  (hxyz : x > 0 ∧ y > 0 ∧ z > 0) : 
  x^a₁ * y^a₂ * z^a₃ + x^a₁ * y^a₃ * z^a₂ + x^a₂ * y^a₁ * z^a₃ + 
  x^a₂ * y^a₃ * z^a₁ + x^a₃ * y^a₁ * z^a₂ + x^a₃ * y^a₂ * z^a₁ ≥ 
  x^b₁ * y^b₂ * z^b₃ + x^b₁ * y^b₃ * z^b₂ + x^b₂ * y^b₁ * z^b₃ + 
  x^b₂ * y^b₃ * z^b₁ + x^b₃ * y^b₁ * z^b₂ + x^b₃ * y^b₂ * z^b₁ :=
sorry

end muirhead_inequality_l76_7658


namespace range_of_f_l76_7677

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3,
  ∃ y ∈ Set.Ico 0 9,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Ico 0 9 :=
sorry

end range_of_f_l76_7677


namespace angle_sum_ninety_degrees_l76_7635

theorem angle_sum_ninety_degrees (A B : Real) (h : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2) :
  A + B = π / 2 := by
  sorry

end angle_sum_ninety_degrees_l76_7635


namespace min_radius_circle_equation_l76_7618

/-- The line on which points A and B move --/
def line (x y : ℝ) : Prop := 3 * x + y - 10 = 0

/-- The circle M with diameter AB --/
def circle_M (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (a.1 + b.1) / 2)^2 + (p.2 - (a.2 + b.2) / 2)^2 = ((a.1 - b.1)^2 + (a.2 - b.2)^2) / 4}

/-- The origin point --/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem stating the standard equation of circle M when its radius is minimum --/
theorem min_radius_circle_equation :
  ∀ a b : ℝ × ℝ,
  line a.1 a.2 → line b.1 b.2 →
  origin ∈ circle_M a b →
  (∀ c d : ℝ × ℝ, line c.1 c.2 → line d.1 d.2 → origin ∈ circle_M c d →
    (a.1 - b.1)^2 + (a.2 - b.2)^2 ≤ (c.1 - d.1)^2 + (c.2 - d.2)^2) →
  circle_M a b = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 10} :=
sorry

end min_radius_circle_equation_l76_7618


namespace chandler_wrapping_paper_sales_l76_7610

def remaining_rolls_to_sell (total_required : ℕ) (sales_to_grandmother : ℕ) (sales_to_uncle : ℕ) (sales_to_neighbor : ℕ) : ℕ :=
  total_required - (sales_to_grandmother + sales_to_uncle + sales_to_neighbor)

theorem chandler_wrapping_paper_sales : 
  remaining_rolls_to_sell 12 3 4 3 = 2 := by
  sorry

end chandler_wrapping_paper_sales_l76_7610


namespace square_area_15cm_l76_7621

/-- The area of a square with side length 15 cm is 225 square centimeters. -/
theorem square_area_15cm (side_length : ℝ) (area : ℝ) : 
  side_length = 15 → area = side_length ^ 2 → area = 225 := by
  sorry

end square_area_15cm_l76_7621


namespace simplify_fraction_l76_7655

theorem simplify_fraction : (18 : ℚ) * (8 / 12) * (1 / 9) * 4 = 16 / 3 := by sorry

end simplify_fraction_l76_7655


namespace willies_stickers_l76_7671

/-- Willie's sticker problem -/
theorem willies_stickers (initial : ℕ) (remaining : ℕ) (given : ℕ) : 
  initial = 36 → remaining = 29 → given = initial - remaining :=
by sorry

end willies_stickers_l76_7671


namespace roots_sum_problem_l76_7667

theorem roots_sum_problem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → b^2 - 5*b + 6 = 0 → a^4 + a^5*b^3 + a^3*b^5 + b^4 = 2905 := by
  sorry

end roots_sum_problem_l76_7667


namespace smallest_three_digit_divisible_l76_7648

theorem smallest_three_digit_divisible : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (6 ∣ n) ∧ (5 ∣ n) ∧ (8 ∣ n) ∧ (9 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (6 ∣ m) ∧ (5 ∣ m) ∧ (8 ∣ m) ∧ (9 ∣ m) → n ≤ m) ∧
  n = 360 :=
by sorry

end smallest_three_digit_divisible_l76_7648


namespace fraction_simplification_l76_7602

theorem fraction_simplification : (5 - 2) / (2 + 1) = 1 := by
  sorry

end fraction_simplification_l76_7602


namespace rectangle_perimeter_l76_7640

/-- Represents the side lengths of squares in ascending order -/
structure SquareSides where
  b₁ : ℝ
  b₂ : ℝ
  b₃ : ℝ
  b₄ : ℝ
  b₅ : ℝ
  b₆ : ℝ
  h_order : 0 < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧ b₅ < b₆

/-- Represents a rectangle partitioned into six squares -/
structure PartitionedRectangle where
  sides : SquareSides
  length : ℝ
  width : ℝ
  h_partition : length = sides.b₃ + sides.b₆ ∧ width = sides.b₁ + sides.b₅
  h_sum_smallest : sides.b₁ + sides.b₂ = sides.b₃
  h_longest_side : 2 * length = 3 * sides.b₆

theorem rectangle_perimeter (rect : PartitionedRectangle) :
    2 * (rect.length + rect.width) = 12 * rect.sides.b₆ := by
  sorry


end rectangle_perimeter_l76_7640


namespace parallel_lines_solution_l76_7601

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line equation: ax + 2y + 6 = 0 -/
def line1 (a x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

/-- The second line equation: x + (a-1)y + (a^2-1) = 0 -/
def line2 (a x y : ℝ) : Prop :=
  x + (a - 1) * y + (a^2 - 1) = 0

/-- The theorem stating that given the two parallel lines, a = -1 -/
theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, parallel_lines a 2 1 (a-1) 1 (a^2-1)) →
  a = -1 :=
by sorry

end parallel_lines_solution_l76_7601


namespace tan_identity_implies_cos_squared_l76_7666

theorem tan_identity_implies_cos_squared (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 4) : 
  Real.cos (θ + π/4)^2 = 1/4 := by
sorry

end tan_identity_implies_cos_squared_l76_7666


namespace johns_change_l76_7647

/-- The change John receives when buying barbells -/
def change_received (num_barbells : ℕ) (barbell_cost : ℕ) (money_given : ℕ) : ℕ :=
  money_given - (num_barbells * barbell_cost)

/-- Theorem: John's change when buying 3 barbells at $270 each and giving $850 is $40 -/
theorem johns_change :
  change_received 3 270 850 = 40 := by
  sorry

end johns_change_l76_7647


namespace ski_trip_sponsorship_l76_7651

/-- The ski trip sponsorship problem -/
theorem ski_trip_sponsorship 
  (total : ℝ) 
  (first_father : ℝ) 
  (second_father third_father fourth_father : ℝ) 
  (h1 : first_father = 11500)
  (h2 : second_father = (1/3) * (total - second_father))
  (h3 : third_father = (1/4) * (total - third_father))
  (h4 : fourth_father = (1/5) * (total - fourth_father))
  (h5 : total = first_father + second_father + third_father + fourth_father) :
  second_father = 7500 ∧ third_father = 6000 ∧ fourth_father = 5000 := by
  sorry

#eval Float.toString 7500
#eval Float.toString 6000
#eval Float.toString 5000

end ski_trip_sponsorship_l76_7651


namespace probability_is_half_l76_7633

/-- A game where a square is divided into triangular sections and some are shaded -/
structure SquareGame where
  total_sections : ℕ
  shaded_sections : ℕ
  h_total : total_sections = 8
  h_shaded : shaded_sections = 4

/-- The probability of landing on a shaded section -/
def probability_shaded (game : SquareGame) : ℚ :=
  game.shaded_sections / game.total_sections

/-- Theorem: The probability of landing on a shaded section is 1/2 -/
theorem probability_is_half (game : SquareGame) : probability_shaded game = 1/2 := by
  sorry

#eval probability_shaded { total_sections := 8, shaded_sections := 4, h_total := rfl, h_shaded := rfl }

end probability_is_half_l76_7633


namespace fraction_order_l76_7683

theorem fraction_order : (25 : ℚ) / 21 < 23 / 19 ∧ 23 / 19 < 21 / 17 := by
  sorry

end fraction_order_l76_7683


namespace insurance_covers_80_percent_l76_7623

def number_of_vaccines : ℕ := 10
def cost_per_vaccine : ℚ := 45
def cost_of_doctors_visit : ℚ := 250
def trip_cost : ℚ := 1200
def toms_payment : ℚ := 1340

def total_medical_cost : ℚ := number_of_vaccines * cost_per_vaccine + cost_of_doctors_visit
def total_trip_cost : ℚ := trip_cost + total_medical_cost
def insurance_coverage : ℚ := total_trip_cost - toms_payment
def insurance_coverage_percentage : ℚ := insurance_coverage / total_medical_cost * 100

theorem insurance_covers_80_percent :
  insurance_coverage_percentage = 80 := by sorry

end insurance_covers_80_percent_l76_7623


namespace simplify_sum_of_powers_l76_7629

theorem simplify_sum_of_powers : 2^2 + 2^2 + 2^2 + 2^2 = 2^4 := by sorry

end simplify_sum_of_powers_l76_7629


namespace ceiling_sum_sqrt_l76_7605

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_sqrt_l76_7605


namespace prob_neither_red_nor_green_l76_7646

/-- Given a bag with green, black, and red pens, this theorem proves the probability
    of picking a pen that is neither red nor green. -/
theorem prob_neither_red_nor_green (green black red : ℕ) 
  (h_green : green = 5) 
  (h_black : black = 6) 
  (h_red : red = 7) : 
  (black : ℚ) / (green + black + red) = 1/3 := by
  sorry

end prob_neither_red_nor_green_l76_7646


namespace tangent_line_to_circle_l76_7645

/-- The value of m for which the line x + y + m = 0 is tangent to the circle x² + y² = m -/
theorem tangent_line_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 ≠ m) ∧ 
  (∃ x y : ℝ, x + y + m = 0 ∧ x^2 + y^2 = m) → 
  m = 2 :=
sorry

end tangent_line_to_circle_l76_7645


namespace circle_power_theorem_l76_7625

structure Circle where
  a : ℝ
  b : ℝ
  R : ℝ

def power (c : Circle) (x₁ y₁ : ℝ) : ℝ :=
  (x₁ - c.a)^2 + (y₁ - c.b)^2 - c.R^2

def distance_squared (x₁ y₁ a b : ℝ) : ℝ :=
  (x₁ - a)^2 + (y₁ - b)^2

theorem circle_power_theorem (c : Circle) (x₁ y₁ : ℝ) :
  -- 1. Power definition
  power c x₁ y₁ = (x₁ - c.a)^2 + (y₁ - c.b)^2 - c.R^2 ∧
  -- 2. Power sign properties
  (distance_squared x₁ y₁ c.a c.b > c.R^2 → power c x₁ y₁ > 0) ∧
  (distance_squared x₁ y₁ c.a c.b < c.R^2 → power c x₁ y₁ < 0) ∧
  (distance_squared x₁ y₁ c.a c.b = c.R^2 → power c x₁ y₁ = 0) ∧
  -- 3. Tangent length property
  (distance_squared x₁ y₁ c.a c.b > c.R^2 → 
    ∃ p, p^2 = power c x₁ y₁ ∧ p ≥ 0) :=
by
  sorry

end circle_power_theorem_l76_7625


namespace line_through_point_parallel_to_line_l76_7619

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.a * l2.b = l2.a * l1.b

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (point : Point) :
  ∃ (result_line : Line),
    parallel result_line given_line ∧
    on_line point result_line ∧
    result_line.a = 1 ∧
    result_line.b = -2 ∧
    result_line.c = -1 :=
by sorry

end line_through_point_parallel_to_line_l76_7619


namespace smallest_positive_solution_l76_7622

theorem smallest_positive_solution (x : ℕ) : 
  (∃ k : ℤ, 45 * x + 15 = 5 + 28 * k) ∧ 
  (∀ y : ℕ, y < x → ¬(∃ k : ℤ, 45 * y + 15 = 5 + 28 * k)) → 
  x = 18 := by sorry

end smallest_positive_solution_l76_7622


namespace problem_statement_l76_7693

theorem problem_statement (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |x| = 2) : 
  x^4 + c*d*x^2 - a - b = 20 := by
sorry

end problem_statement_l76_7693


namespace exactly_two_more_heads_probability_l76_7627

/-- The number of coins being flipped -/
def num_coins : ℕ := 10

/-- The number of heads required to have exactly two more heads than tails -/
def required_heads : ℕ := (num_coins + 2) / 2

/-- The probability of getting heads on a single fair coin flip -/
def prob_heads : ℚ := 1 / 2

theorem exactly_two_more_heads_probability :
  (Nat.choose num_coins required_heads : ℚ) * prob_heads ^ required_heads * (1 - prob_heads) ^ (num_coins - required_heads) = 210 / 1024 := by
  sorry

end exactly_two_more_heads_probability_l76_7627


namespace rhombus_area_l76_7680

/-- The area of a rhombus with side length 4 cm and an interior angle of 45 degrees is 8√2 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) : 
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end rhombus_area_l76_7680


namespace students_liking_both_sports_and_music_l76_7624

theorem students_liking_both_sports_and_music
  (total : ℕ)
  (sports : ℕ)
  (music : ℕ)
  (neither : ℕ)
  (h_total : total = 55)
  (h_sports : sports = 43)
  (h_music : music = 34)
  (h_neither : neither = 4) :
  ∃ (both : ℕ), both = sports + music - total + neither ∧ both = 26 :=
by sorry

end students_liking_both_sports_and_music_l76_7624


namespace teal_color_perception_l76_7603

theorem teal_color_perception (total : ℕ) (greenish : ℕ) (both : ℕ) (neither : ℕ) :
  total = 120 →
  greenish = 80 →
  both = 35 →
  neither = 20 →
  ∃ bluish : ℕ, bluish = 55 ∧ bluish = total - (greenish - both) - both - neither :=
by sorry

end teal_color_perception_l76_7603


namespace sixth_power_sum_l76_7672

theorem sixth_power_sum (r : ℝ) (h : (r + 1/r)^4 = 17) : 
  r^6 + 1/r^6 = Real.sqrt 17 - 6 := by
  sorry

end sixth_power_sum_l76_7672


namespace bag_probability_l76_7615

theorem bag_probability (d x : ℕ) : 
  d = x + (x + 1) + (x + 2) →
  (x : ℚ) / d < 1 / 6 →
  d = 3 := by
sorry

end bag_probability_l76_7615


namespace rectangular_field_fencing_l76_7634

theorem rectangular_field_fencing (area : ℝ) (uncovered_side : ℝ) : 
  area = 210 → uncovered_side = 20 → 
  2 * (area / uncovered_side) + uncovered_side = 41 := by
  sorry

end rectangular_field_fencing_l76_7634


namespace horner_V₃_eq_71_l76_7691

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Coefficients of the polynomial f(x) = 2x⁶ + 5x⁵ + 6x⁴ + 23x³ - 8x² + 10x - 3 -/
def f_coeffs : List ℚ := [2, 5, 6, 23, -8, 10, -3]

/-- The value of x -/
def x : ℚ := 2

/-- V₃ in Horner's method -/
def V₃ : ℚ := 
  let v₀ : ℚ := f_coeffs[0]!
  let v₁ : ℚ := v₀ * x + f_coeffs[1]!
  let v₂ : ℚ := v₁ * x + f_coeffs[2]!
  v₂ * x + f_coeffs[3]!

theorem horner_V₃_eq_71 : V₃ = 71 := by
  sorry

#eval V₃

end horner_V₃_eq_71_l76_7691


namespace inequality_system_equivalence_l76_7656

theorem inequality_system_equivalence :
  ∀ x : ℝ, (x + 1 ≥ 2 ∧ x > 0) ↔ x ≥ 1 := by
  sorry

end inequality_system_equivalence_l76_7656


namespace patio_tiles_l76_7649

theorem patio_tiles (c : ℕ) (h1 : c > 2) : 
  c * 10 = (c - 2) * (10 + 4) → c * 10 = 70 := by
  sorry

#check patio_tiles

end patio_tiles_l76_7649


namespace equation_solution_l76_7654

theorem equation_solution :
  ∃ x : ℚ, -8 * (2 - x)^3 = 27 ∧ x = 5/2 := by
  sorry

end equation_solution_l76_7654


namespace even_plus_abs_odd_is_even_l76_7696

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem even_plus_abs_odd_is_even
  (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsEven (fun x ↦ f x + |g x|) := by
  sorry

end even_plus_abs_odd_is_even_l76_7696


namespace paper_piles_theorem_l76_7690

theorem paper_piles_theorem (n : ℕ) :
  1000 < n ∧ n < 2000 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) →
  ∃ m : ℕ, m = 41 ∧ m ≠ 1 ∧ m ≠ n ∧ n % m = 0 :=
by sorry

end paper_piles_theorem_l76_7690


namespace min_value_of_exponential_sum_l76_7609

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ (m : ℝ), m = 4 ∧ ∀ x y : ℝ, x + y = 2 → 2^x + 2^y ≥ m :=
sorry

end min_value_of_exponential_sum_l76_7609


namespace smallest_number_with_remainders_l76_7678

theorem smallest_number_with_remainders (n : ℕ) :
  (∃ m : ℕ, n = 5 * m + 4) ∧
  (∃ m : ℕ, n = 6 * m + 5) ∧
  (((∃ m : ℕ, n = 7 * m + 6) → n ≥ 209) ∧
   ((∃ m : ℕ, n = 8 * m + 7) → n ≥ 119)) ∧
  (n = 209 ∨ n = 119) :=
by sorry

end smallest_number_with_remainders_l76_7678


namespace y_coordinate_product_l76_7608

/-- The product of y-coordinates for points on x = -2 that are 12 units from (6, 3) -/
theorem y_coordinate_product : ∃ y₁ y₂ : ℝ,
  ((-2 - 6)^2 + (y₁ - 3)^2 = 12^2) ∧
  ((-2 - 6)^2 + (y₂ - 3)^2 = 12^2) ∧
  y₁ ≠ y₂ ∧
  y₁ * y₂ = -71 := by
sorry

end y_coordinate_product_l76_7608


namespace candy_box_total_l76_7673

theorem candy_box_total (purple orange yellow total : ℕ) : 
  purple + orange + yellow = total →
  2 * orange = 4 * purple →
  5 * purple = 2 * yellow →
  yellow = 40 →
  total = 88 := by
sorry

end candy_box_total_l76_7673


namespace g_range_l76_7674

/-- The function representing the curve C -/
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 3)

/-- The function g(t) representing the product of magnitudes of OP and OQ -/
def g (t : ℝ) : ℝ := |(3 - t) * (1 + t^2)|

/-- Theorem stating that the range of g(t) is [0, ∞) -/
theorem g_range : Set.range g = Set.Ici (0 : ℝ) := by sorry

end g_range_l76_7674


namespace min_value_theorem_l76_7643

/-- The line equation ax - 2by = 2 passes through the center of the circle x² + y² - 4x + 2y + 1 = 0 -/
def line_passes_through_center (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 ∧ a*x - 2*b*y = 2

/-- The minimum value of 1/a + 1/b + 1/(ab) given the conditions -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_center : line_passes_through_center a b) : 
  (1/a + 1/b + 1/(a*b)) ≥ 8 := by
  sorry

end min_value_theorem_l76_7643


namespace garden_furniture_costs_l76_7630

theorem garden_furniture_costs (total_cost bench_cost table_cost umbrella_cost : ℝ) :
  total_cost = 765 ∧
  table_cost = 2 * bench_cost ∧
  umbrella_cost = 3 * bench_cost →
  bench_cost = 127.5 ∧
  table_cost = 255 ∧
  umbrella_cost = 382.5 :=
by sorry

end garden_furniture_costs_l76_7630


namespace total_cost_is_1027_2_l76_7606

/-- The cost relationship between mangos, rice, and flour -/
structure CostRelationship where
  mango_cost : ℝ  -- Cost per kg of mangos
  rice_cost : ℝ   -- Cost per kg of rice
  flour_cost : ℝ  -- Cost per kg of flour
  mango_rice_relation : 10 * mango_cost = 24 * rice_cost
  flour_rice_relation : 6 * flour_cost = 2 * rice_cost
  flour_cost_value : flour_cost = 24

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost (cr : CostRelationship) : ℝ :=
  4 * cr.mango_cost + 3 * cr.rice_cost + 5 * cr.flour_cost

/-- Theorem stating that the total cost is $1027.2 -/
theorem total_cost_is_1027_2 (cr : CostRelationship) :
  total_cost cr = 1027.2 := by
  sorry

#check total_cost_is_1027_2

end total_cost_is_1027_2_l76_7606


namespace meaningful_fraction_l76_7665

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 2023)) ↔ x ≠ 2023 := by sorry

end meaningful_fraction_l76_7665


namespace probability_four_even_dice_l76_7604

theorem probability_four_even_dice (n : ℕ) (p : ℚ) : 
  n = 8 →
  p = 1/2 →
  (n.choose (n/2)) * p^n = 35/128 :=
by sorry

end probability_four_even_dice_l76_7604


namespace red_light_probability_l76_7669

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of seeing a red light given the traffic light durations -/
def probability_red_light (d : TrafficLightDuration) : ℚ :=
  d.red / (d.red + d.yellow + d.green)

/-- Theorem: The probability of seeing a red light is 2/5 for the given durations -/
theorem red_light_probability (d : TrafficLightDuration) 
  (h_red : d.red = 30)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 40) : 
  probability_red_light d = 2/5 := by
  sorry

#eval probability_red_light ⟨30, 5, 40⟩

end red_light_probability_l76_7669


namespace quadratic_equations_solution_l76_7689

theorem quadratic_equations_solution (m n k : ℝ) : 
  (∃ x : ℝ, m * x^2 + n = 0) ∧
  (∃ x : ℝ, n * x^2 + k = 0) ∧
  (∃ x : ℝ, k * x^2 + m = 0) →
  m = 0 ∧ n = 0 ∧ k = 0 := by
sorry

end quadratic_equations_solution_l76_7689


namespace laptop_gifting_l76_7662

theorem laptop_gifting (n m : ℕ) (hn : n = 15) (hm : m = 3) :
  (n.factorial / (n - m).factorial) = 2730 := by
  sorry

end laptop_gifting_l76_7662


namespace billy_score_is_13_l76_7644

/-- Represents a contestant's performance on the AMC 8 contest -/
structure AMC8Performance where
  total_questions : Nat
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered : Nat
  correct_point_value : Nat
  incorrect_point_value : Nat
  unanswered_point_value : Nat

/-- Calculates the score for an AMC 8 performance -/
def calculate_score (performance : AMC8Performance) : Nat :=
  performance.correct_answers * performance.correct_point_value +
  performance.incorrect_answers * performance.incorrect_point_value +
  performance.unanswered * performance.unanswered_point_value

/-- Billy's performance on the AMC 8 contest -/
def billy_performance : AMC8Performance := {
  total_questions := 25,
  correct_answers := 13,
  incorrect_answers := 7,
  unanswered := 5,
  correct_point_value := 1,
  incorrect_point_value := 0,
  unanswered_point_value := 0
}

theorem billy_score_is_13 : calculate_score billy_performance = 13 := by
  sorry

end billy_score_is_13_l76_7644


namespace sine_function_omega_l76_7687

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0, 
    if f(π/6) = f(π/3) and f(x) has a maximum value but no minimum value 
    in the interval (π/6, π/3), then ω = 2/3 -/
theorem sine_function_omega (ω : ℝ) (h_pos : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 3)
  (f (π / 6) = f (π / 3)) → 
  (∃ (x : ℝ), x ∈ Set.Ioo (π / 6) (π / 3) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo (π / 6) (π / 3) → f y ≤ f x)) →
  (∀ (x : ℝ), x ∈ Set.Ioo (π / 6) (π / 3) → 
    ∃ (y : ℝ), y ∈ Set.Ioo (π / 6) (π / 3) ∧ f y < f x) →
  ω = 2 / 3 := by
sorry

end sine_function_omega_l76_7687


namespace train_length_l76_7659

/-- The length of a train given its speed and time to cross a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 64 * (5 / 18) → time = 9 → speed * time = 160 := by
  sorry

end train_length_l76_7659


namespace new_average_age_l76_7668

theorem new_average_age (n : ℕ) (original_avg : ℝ) (new_person_age : ℝ) :
  n = 9 ∧ original_avg = 15 ∧ new_person_age = 35 →
  (n * original_avg + new_person_age) / (n + 1) = 17 := by
  sorry

end new_average_age_l76_7668


namespace circles_intersection_tangent_equality_points_l76_7620

-- Define the circles and ellipse
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 6*y + 32 = 0
def C2 (a x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*(8-a)*y + 4*a + 12 = 0
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Theorem for part (I)
theorem circles_intersection :
  ∀ a : ℝ, C1 4 2 ∧ C1 6 4 ∧ C2 a 4 2 ∧ C2 a 6 4 := by sorry

-- Theorem for part (II)
theorem tangent_equality_points :
  ∀ x y : ℝ, Ellipse x y →
    (∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧
      (x^2 + y^2 - 10*x - 6*y + 32 = x^2 + y^2 - 2*a₁*x - 2*(8-a₁)*y + 4*a₁ + 12) ∧
      (x^2 + y^2 - 10*x - 6*y + 32 = x^2 + y^2 - 2*a₂*x - 2*(8-a₂)*y + 4*a₂ + 12)) ↔
    ((x = 2 ∧ y = 0) ∨ (x = 6/5 ∧ y = -4/5)) := by sorry

end circles_intersection_tangent_equality_points_l76_7620


namespace linear_dependence_implies_k₁_plus_4k₃_eq_zero_l76_7661

def a₁ : Fin 2 → ℝ := ![1, 0]
def a₂ : Fin 2 → ℝ := ![1, -1]
def a₃ : Fin 2 → ℝ := ![2, 2]

theorem linear_dependence_implies_k₁_plus_4k₃_eq_zero :
  ∃ (k₁ k₂ k₃ : ℝ), (k₁ ≠ 0 ∨ k₂ ≠ 0 ∨ k₃ ≠ 0) ∧
    (∀ i : Fin 2, k₁ * a₁ i + k₂ * a₂ i + k₃ * a₃ i = 0) →
  ∀ (k₁ k₂ k₃ : ℝ), (∀ i : Fin 2, k₁ * a₁ i + k₂ * a₂ i + k₃ * a₃ i = 0) →
  k₁ + 4 * k₃ = 0 :=
by sorry

end linear_dependence_implies_k₁_plus_4k₃_eq_zero_l76_7661


namespace aluminum_carbonate_molecular_weight_l76_7686

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of the given moles of Aluminum carbonate in grams -/
def given_molecular_weight : ℝ := 1170

/-- The molecular formula of Aluminum carbonate: Al₂(CO₃)₃ -/
structure AluminumCarbonate where
  Al : Nat
  C : Nat
  O : Nat

/-- The correct molecular formula of Aluminum carbonate -/
def Al2CO3_3 : AluminumCarbonate := ⟨2, 3, 9⟩

/-- Calculate the molecular weight of Aluminum carbonate -/
def molecular_weight (formula : AluminumCarbonate) : ℝ :=
  formula.Al * atomic_weight_Al + formula.C * atomic_weight_C + formula.O * atomic_weight_O

/-- Theorem: The molecular weight of Aluminum carbonate is approximately 234.99 g/mol -/
theorem aluminum_carbonate_molecular_weight :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |molecular_weight Al2CO3_3 - 234.99| < ε :=
sorry

end aluminum_carbonate_molecular_weight_l76_7686


namespace base_subtraction_proof_l76_7694

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_subtraction_proof :
  let base7_num := base_to_decimal [5, 4, 3, 2, 1, 0] 7
  let base8_num := base_to_decimal [4, 5, 3, 2, 1] 8
  base7_num - base8_num = 75620 := by
sorry

end base_subtraction_proof_l76_7694


namespace min_value_of_expression_l76_7685

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) :
  ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0),
    (∀ (u v : ℝ) (hu : u > 0) (hv : v > 0),
      Real.sqrt 3 = Real.sqrt (3^u * 3^(2*v)) →
      2/u + 1/v ≥ 2/x + 1/y) ∧
    2/x + 1/y = 8 :=
by sorry

end min_value_of_expression_l76_7685


namespace folded_paper_properties_l76_7688

/-- Represents a folded rectangular paper with specific properties -/
structure FoldedPaper where
  short_edge : ℝ
  long_edge : ℝ
  fold_length : ℝ
  congruent_triangles : Prop

/-- Theorem stating the properties of the folded paper -/
theorem folded_paper_properties (paper : FoldedPaper) 
  (h1 : paper.short_edge = 12)
  (h2 : paper.long_edge = 18)
  (h3 : paper.congruent_triangles)
  : paper.fold_length = 10 := by
  sorry

end folded_paper_properties_l76_7688


namespace equation_solution_l76_7611

theorem equation_solution : ∃! n : ℚ, (1 : ℚ) / (n + 1) + (2 : ℚ) / (n + 1) + n / (n + 1) = 3 := by
  sorry

end equation_solution_l76_7611


namespace last_four_digits_5_2011_l76_7675

-- Define a function to get the last four digits of a number
def lastFourDigits (n : ℕ) : ℕ := n % 10000

-- Define the cycle length of the last four digits of powers of 5
def cycleLengthPowersOf5 : ℕ := 4

-- Theorem statement
theorem last_four_digits_5_2011 :
  lastFourDigits (5^2011) = lastFourDigits (5^7) :=
by
  sorry

#eval lastFourDigits (5^7)  -- This should output 8125

end last_four_digits_5_2011_l76_7675


namespace cos_2x_value_l76_7626

theorem cos_2x_value (x : ℝ) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : 
  Real.cos (2 * x) = 1/2 := by
  sorry

end cos_2x_value_l76_7626


namespace tape_length_theorem_l76_7652

/-- Given 15 sheets of tape, each 25 cm long, overlapping by 0.5 cm,
    the total length of the attached tape is 3.68 meters. -/
theorem tape_length_theorem (num_sheets : ℕ) (sheet_length : ℝ) (overlap : ℝ) :
  num_sheets = 15 →
  sheet_length = 25 →
  overlap = 0.5 →
  (num_sheets * sheet_length - (num_sheets - 1) * overlap) / 100 = 3.68 := by
  sorry

end tape_length_theorem_l76_7652


namespace quantity_count_l76_7600

theorem quantity_count (total_sum : ℝ) (total_count : ℕ) 
  (subset1_sum : ℝ) (subset1_count : ℕ) 
  (subset2_sum : ℝ) (subset2_count : ℕ) 
  (h1 : total_sum / total_count = 12)
  (h2 : subset1_sum / subset1_count = 4)
  (h3 : subset2_sum / subset2_count = 24)
  (h4 : subset1_count = 3)
  (h5 : subset2_count = 2)
  (h6 : total_sum = subset1_sum + subset2_sum)
  (h7 : total_count = subset1_count + subset2_count) : 
  total_count = 5 := by
sorry


end quantity_count_l76_7600


namespace lottery_winning_probability_l76_7636

/-- Represents the lottery with MegaBall, WinnerBalls, and BonusBall -/
structure Lottery where
  megaBallCount : ℕ
  winnerBallCount : ℕ
  bonusBallCount : ℕ
  winnerBallsPicked : ℕ

/-- Calculates the probability of winning the lottery -/
def winningProbability (l : Lottery) : ℚ :=
  1 / (l.megaBallCount * (l.winnerBallCount.choose l.winnerBallsPicked) * l.bonusBallCount)

/-- The specific lottery configuration -/
def ourLottery : Lottery :=
  { megaBallCount := 30
    winnerBallCount := 50
    bonusBallCount := 15
    winnerBallsPicked := 5 }

/-- Theorem stating the probability of winning our specific lottery -/
theorem lottery_winning_probability :
    winningProbability ourLottery = 1 / 953658000 := by
  sorry

#eval winningProbability ourLottery

end lottery_winning_probability_l76_7636


namespace algebraic_operation_equality_l76_7637

theorem algebraic_operation_equality (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := by
  sorry

end algebraic_operation_equality_l76_7637


namespace absolute_value_ratio_l76_7657

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 12*a*b) :
  |((a+b)/(a-b))| = Real.sqrt (7/5) := by
  sorry

end absolute_value_ratio_l76_7657


namespace range_of_t_l76_7676

def A (t : ℝ) : Set ℝ := {1, t}

theorem range_of_t (t : ℝ) : t ∈ {x : ℝ | x ≠ 1} ↔ t ∈ A t := by
  sorry

end range_of_t_l76_7676


namespace recommendation_plans_count_l76_7670

/-- Represents the number of recommendation spots for each language --/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the gender distribution of candidates --/
structure CandidateGenders :=
  (males : Nat)
  (females : Nat)

/-- Calculates the number of different recommendation plans --/
def count_recommendation_plans (spots : RecommendationSpots) (genders : CandidateGenders) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem recommendation_plans_count :
  let spots := RecommendationSpots.mk 2 2 1
  let genders := CandidateGenders.mk 3 2
  count_recommendation_plans spots genders = 24 := by
  sorry

end recommendation_plans_count_l76_7670


namespace length_of_A_prime_B_prime_l76_7628

/-- Given points A, B, C, and the conditions for A' and B', prove that |A'B'| = 5√2 -/
theorem length_of_A_prime_B_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (∃ t : ℝ, A + t • (A' - A) = C) →
  (∃ s : ℝ, B + s • (B' - B) = C) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 5 * Real.sqrt 2 :=
by sorry

end length_of_A_prime_B_prime_l76_7628


namespace complex_fraction_simplification_l76_7607

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 → (11 + 3*i) / (1 - 2*i) = 1 + 5*i := by
  sorry

end complex_fraction_simplification_l76_7607


namespace problem_statement_l76_7653

theorem problem_statement : (2112 - 2021)^2 / 169 = 49 := by sorry

end problem_statement_l76_7653


namespace max_halls_visited_l76_7692

structure Museum :=
  (total_halls : ℕ)
  (painting_halls : ℕ)
  (sculpture_halls : ℕ)
  (is_even : total_halls % 2 = 0)
  (half_paintings : painting_halls = total_halls / 2)
  (half_sculptures : sculpture_halls = total_halls / 2)

def alternating_tour (m : Museum) (start_painting : Bool) (end_painting : Bool) : ℕ → Prop
  | 0 => start_painting
  | 1 => ¬start_painting
  | (n+2) => alternating_tour m start_painting end_painting n

theorem max_halls_visited 
  (m : Museum) 
  (h : m.total_halls = 16) 
  (start_painting : Bool) 
  (end_painting : Bool) 
  (h_start_end : start_painting = end_painting) :
  ∃ (n : ℕ), n ≤ m.total_halls - 1 ∧ 
    alternating_tour m start_painting end_painting n ∧ 
    ∀ (k : ℕ), alternating_tour m start_painting end_painting k → k ≤ n :=
sorry

end max_halls_visited_l76_7692
