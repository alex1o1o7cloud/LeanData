import Mathlib

namespace NUMINAMATH_GPT_total_height_correct_l241_24172

-- Stack and dimensions setup
def height_of_disc_stack (top_diameter bottom_diameter disc_thickness : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  num_discs * disc_thickness

def total_height (top_diameter bottom_diameter disc_thickness cylinder_height : ℕ) : ℕ :=
  height_of_disc_stack top_diameter bottom_diameter disc_thickness + cylinder_height

-- Given conditions
def top_diameter := 15
def bottom_diameter := 1
def disc_thickness := 2
def cylinder_height := 10
def correct_answer := 26

-- Proof problem
theorem total_height_correct :
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = correct_answer :=
by
  sorry

end NUMINAMATH_GPT_total_height_correct_l241_24172


namespace NUMINAMATH_GPT_chord_length_condition_l241_24128

theorem chord_length_condition (c : ℝ) (h : c > 0) :
  (∃ (x1 x2 : ℝ), 
    x1 ≠ x2 ∧ 
    dist (x1, x1^2) (x2, x2^2) = 2 ∧ 
    ∃ k : ℝ, x1 * k + c = x1^2 ∧ x2 * k + c = x2^2 ) 
    ↔ c > 0 :=
sorry

end NUMINAMATH_GPT_chord_length_condition_l241_24128


namespace NUMINAMATH_GPT_bowling_ball_weight_l241_24105

-- Definitions for the conditions
def kayak_weight : ℕ := 36
def total_weight_of_two_kayaks := 2 * kayak_weight
def total_weight_of_nine_bowling_balls (ball_weight : ℕ) := 9 * ball_weight  

theorem bowling_ball_weight (w : ℕ) (h1 : total_weight_of_two_kayaks = total_weight_of_nine_bowling_balls w) : w = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l241_24105


namespace NUMINAMATH_GPT_coefficient_of_squared_term_l241_24188

theorem coefficient_of_squared_term (a b c : ℝ) (h_eq : 5 * a^2 + 14 * b + 5 = 0) :
  a = 5 :=
sorry

end NUMINAMATH_GPT_coefficient_of_squared_term_l241_24188


namespace NUMINAMATH_GPT_find_b_l241_24184

noncomputable def ellipse_foci (a b : ℝ) (hb : b > 0) (hab : a > b) : Prop :=
∃ (F1 F2 P : ℝ×ℝ), 
    (∃ (h : a > b), (2 * b^2 + 9 = a^2)) ∧ 
    (dist P F1 + dist P F2 = 2 * a) ∧ 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (2 * 4 * (a^2 - b^2) = 36)

theorem find_b (a b : ℝ) (hb : b > 0) (hab : a > b) : 
    ellipse_foci a b hb hab → b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l241_24184


namespace NUMINAMATH_GPT_find_value_of_y_l241_24162

theorem find_value_of_y (y : ℚ) (h : 3 * y / 7 = 14) : y = 98 / 3 := 
by
  /- Proof to be completed -/
  sorry

end NUMINAMATH_GPT_find_value_of_y_l241_24162


namespace NUMINAMATH_GPT_gcd_lcm_sum_l241_24135

theorem gcd_lcm_sum :
  ∀ (a b c d : ℕ), gcd a b + lcm c d = 74 :=
by
  let a := 42
  let b := 70
  let c := 20
  let d := 15
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l241_24135


namespace NUMINAMATH_GPT_sum_of_n_plus_k_l241_24183

theorem sum_of_n_plus_k (n k : ℕ) (h1 : 2 * (n - k) = 3 * (k + 1)) (h2 : 3 * (n - k - 1) = 4 * (k + 2)) : n + k = 47 := by
  sorry

end NUMINAMATH_GPT_sum_of_n_plus_k_l241_24183


namespace NUMINAMATH_GPT_pages_removed_iff_original_pages_l241_24185

def booklet_sum (n r : ℕ) : ℕ :=
  (n * (2 * n + 1)) - (4 * r - 1)

theorem pages_removed_iff_original_pages (n r : ℕ) :
  booklet_sum n r = 963 ↔ (2 * n = 44 ∧ (2 * r - 1, 2 * r) = (13, 14)) :=
sorry

end NUMINAMATH_GPT_pages_removed_iff_original_pages_l241_24185


namespace NUMINAMATH_GPT_divide_friends_among_teams_l241_24182

theorem divide_friends_among_teams :
  let friends_num := 8
  let teams_num := 4
  (teams_num ^ friends_num) = 65536 := by
  sorry

end NUMINAMATH_GPT_divide_friends_among_teams_l241_24182


namespace NUMINAMATH_GPT_zero_in_interval_l241_24118

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_in_interval : ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The correct statement only
  sorry

end NUMINAMATH_GPT_zero_in_interval_l241_24118


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l241_24146

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneq1 : a ≠ b) (hneq2 : b ≠ c) (hneq3 : a ≠ c) (hvol : a * b * c = 770) : 2 * (a * b + b * c + c * a) = 1098 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l241_24146


namespace NUMINAMATH_GPT_walker_rate_l241_24155

theorem walker_rate (W : ℝ) :
  (∀ t : ℝ, t = 5 / 60 ∧ t = 20 / 60 → 20 * t = (5 * 20 / 3) ∧ W * (1 / 3) = 5 / 3) →
  W = 5 :=
by
  sorry

end NUMINAMATH_GPT_walker_rate_l241_24155


namespace NUMINAMATH_GPT_smallest_value_of_x_l241_24119

theorem smallest_value_of_x (x : ℝ) (hx : |3 * x + 7| = 26) : x = -11 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_x_l241_24119


namespace NUMINAMATH_GPT_sandbox_side_length_l241_24115

theorem sandbox_side_length (side_length : ℝ) (sand_sq_inches_per_pound : ℝ := 80 / 30) (total_sand_pounds : ℝ := 600) :
  (side_length ^ 2 = total_sand_pounds * sand_sq_inches_per_pound) → side_length = 40 := 
by
  sorry

end NUMINAMATH_GPT_sandbox_side_length_l241_24115


namespace NUMINAMATH_GPT_find_number_l241_24179

theorem find_number (x : ℝ) (h : 4 * (3 * x / 5 - 220) = 320) : x = 500 :=
sorry

end NUMINAMATH_GPT_find_number_l241_24179


namespace NUMINAMATH_GPT_jean_grandchildren_total_giveaway_l241_24131

theorem jean_grandchildren_total_giveaway :
  let num_grandchildren := 3
  let cards_per_grandchild_per_year := 2
  let amount_per_card := 80
  let total_amount_per_grandchild_per_year := cards_per_grandchild_per_year * amount_per_card
  let total_amount_per_year := num_grandchildren * total_amount_per_grandchild_per_year
  total_amount_per_year = 480 :=
by
  sorry

end NUMINAMATH_GPT_jean_grandchildren_total_giveaway_l241_24131


namespace NUMINAMATH_GPT_value_of_x4_plus_1_div_x4_l241_24175

theorem value_of_x4_plus_1_div_x4 (x : ℝ) (hx : x^2 + 1 / x^2 = 2) : x^4 + 1 / x^4 = 2 := 
sorry

end NUMINAMATH_GPT_value_of_x4_plus_1_div_x4_l241_24175


namespace NUMINAMATH_GPT_megatech_budget_allocation_l241_24107

theorem megatech_budget_allocation :
  let microphotonics := 14
  let food_additives := 10
  let gmo := 24
  let industrial_lubricants := 8
  let basic_astrophysics := 25
  microphotonics + food_additives + gmo + industrial_lubricants + basic_astrophysics = 81 →
  100 - 81 = 19 :=
by
  intros
  -- We are given the sums already, so directly calculate the remaining percentage.
  sorry

end NUMINAMATH_GPT_megatech_budget_allocation_l241_24107


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l241_24132

theorem area_of_triangle_ABC : 
  let A := (1, 1)
  let B := (4, 1)
  let C := (1, 5)
  let area := 6
  (1:ℝ) * abs (1 * (1 - 5) + 4 * (5 - 1) + 1 * (1 - 1)) / 2 = area := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l241_24132


namespace NUMINAMATH_GPT_possible_values_of_m_l241_24137

theorem possible_values_of_m (m : ℝ) :
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  (∀ x, x ∈ B → x ∈ A) ↔ m = 0 ∨ m = -1 ∨ m = -1 / 3 :=
by
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  sorry -- Proof needed

end NUMINAMATH_GPT_possible_values_of_m_l241_24137


namespace NUMINAMATH_GPT_mary_total_spent_l241_24101

def store1_shirt : ℝ := 13.04
def store1_jacket : ℝ := 12.27
def store2_shoes : ℝ := 44.15
def store2_dress : ℝ := 25.50
def hat_price : ℝ := 9.99
def discount : ℝ := 0.10
def store4_handbag : ℝ := 30.93
def store4_scarf : ℝ := 7.42
def sunglasses_price : ℝ := 20.75
def sales_tax : ℝ := 0.05

def store1_total : ℝ := store1_shirt + store1_jacket
def store2_total : ℝ := store2_shoes + store2_dress
def store3_total : ℝ := 
  let hat_cost := hat_price * 2
  let discount_amt := hat_cost * discount
  hat_cost - discount_amt
def store4_total : ℝ := store4_handbag + store4_scarf
def store5_total : ℝ := 
  let tax := sunglasses_price * sales_tax
  sunglasses_price + tax

def total_spent : ℝ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_total_spent : total_spent = 173.08 := sorry

end NUMINAMATH_GPT_mary_total_spent_l241_24101


namespace NUMINAMATH_GPT_accommodation_arrangements_l241_24166

-- Given conditions
def triple_room_capacity : Nat := 3
def double_room_capacity : Nat := 2
def single_room_capacity : Nat := 1
def num_adult_men : Nat := 4
def num_little_boys : Nat := 2

-- Ensuring little boys are always accompanied by an adult and all rooms are occupied
def is_valid_arrangement (triple double single : Nat × Nat) : Prop :=
  let (triple_adults, triple_boys) := triple
  let (double_adults, double_boys) := double
  let (single_adults, single_boys) := single
  triple_adults + double_adults + single_adults = num_adult_men ∧
  triple_boys + double_boys + single_boys = num_little_boys ∧
  triple = (triple_room_capacity, num_little_boys) ∨
  (triple = (triple_room_capacity, 1) ∧ double = (double_room_capacity, 1)) ∧
  triple_adults + triple_boys = triple_room_capacity ∧
  double_adults + double_boys = double_room_capacity ∧
  single_adults + single_boys = single_room_capacity

-- Main theorem statement
theorem accommodation_arrangements : ∃ (triple double single : Nat × Nat),
  is_valid_arrangement triple double single ∧
  -- The number 36 comes from the correct answer in the solution steps part b)
  (triple.1 + double.1 + single.1 = 4 ∧ triple.2 + double.2 + single.2 = 2) :=
sorry

end NUMINAMATH_GPT_accommodation_arrangements_l241_24166


namespace NUMINAMATH_GPT_pow_zero_eq_one_l241_24125

theorem pow_zero_eq_one : (-2023)^0 = 1 :=
by
  -- The proof of this theorem will go here.
  sorry

end NUMINAMATH_GPT_pow_zero_eq_one_l241_24125


namespace NUMINAMATH_GPT_compare_abc_l241_24102

noncomputable def a : ℝ := Real.log 10 / Real.log 5
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem compare_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_compare_abc_l241_24102


namespace NUMINAMATH_GPT_initial_employees_l241_24133

theorem initial_employees (E : ℕ)
  (salary_per_employee : ℕ)
  (laid_off_fraction : ℚ)
  (total_paid_remaining : ℕ)
  (remaining_employees : ℕ) :
  salary_per_employee = 2000 →
  laid_off_fraction = 1 / 3 →
  total_paid_remaining = 600000 →
  remaining_employees = total_paid_remaining / salary_per_employee →
  (2 / 3 : ℚ) * E = remaining_employees →
  E = 450 := by
  sorry

end NUMINAMATH_GPT_initial_employees_l241_24133


namespace NUMINAMATH_GPT_whipped_cream_needed_l241_24138

def total_days : ℕ := 15
def odd_days_count : ℕ := 8
def even_days_count : ℕ := 7

def pumpkin_pies_on_odd_days : ℕ := 3 * odd_days_count
def apple_pies_on_odd_days : ℕ := 2 * odd_days_count

def pumpkin_pies_on_even_days : ℕ := 2 * even_days_count
def apple_pies_on_even_days : ℕ := 4 * even_days_count

def total_pumpkin_pies_baked : ℕ := pumpkin_pies_on_odd_days + pumpkin_pies_on_even_days
def total_apple_pies_baked : ℕ := apple_pies_on_odd_days + apple_pies_on_even_days

def tiffany_pumpkin_pies_consumed : ℕ := 2
def tiffany_apple_pies_consumed : ℕ := 5

def remaining_pumpkin_pies : ℕ := total_pumpkin_pies_baked - tiffany_pumpkin_pies_consumed
def remaining_apple_pies : ℕ := total_apple_pies_baked - tiffany_apple_pies_consumed

def whipped_cream_for_pumpkin_pies : ℕ := 2 * remaining_pumpkin_pies
def whipped_cream_for_apple_pies : ℕ := remaining_apple_pies

def total_whipped_cream_needed : ℕ := whipped_cream_for_pumpkin_pies + whipped_cream_for_apple_pies

theorem whipped_cream_needed : total_whipped_cream_needed = 111 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_whipped_cream_needed_l241_24138


namespace NUMINAMATH_GPT_min_value_function_l241_24196

theorem min_value_function (x : ℝ) (h : 1 < x) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y ≥ 3) :=
sorry

end NUMINAMATH_GPT_min_value_function_l241_24196


namespace NUMINAMATH_GPT_find_sum_of_cubes_l241_24198

noncomputable def roots (a b c : ℝ) : Prop :=
  5 * a^3 + 2014 * a + 4027 = 0 ∧ 
  5 * b^3 + 2014 * b + 4027 = 0 ∧ 
  5 * c^3 + 2014 * c + 4027 = 0

theorem find_sum_of_cubes (a b c : ℝ) (h : roots a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 :=
sorry

end NUMINAMATH_GPT_find_sum_of_cubes_l241_24198


namespace NUMINAMATH_GPT_shortest_distance_between_circles_l241_24164

theorem shortest_distance_between_circles :
  let circle1 := (x^2 - 12*x + y^2 - 6*y + 9 = 0)
  let circle2 := (x^2 + 10*x + y^2 + 8*y + 34 = 0)
  -- Centers and radii from conditions above:
  let center1 := (6, 3)
  let radius1 := 3
  let center2 := (-5, -4)
  let radius2 := Real.sqrt 7
  let distance_centers := Real.sqrt ((6 - (-5))^2 + (3 - (-4))^2)
  -- Calculate shortest distance
  distance_centers - (radius1 + radius2) = Real.sqrt 170 - 3 - Real.sqrt 7 := sorry

end NUMINAMATH_GPT_shortest_distance_between_circles_l241_24164


namespace NUMINAMATH_GPT_fraction_ratio_l241_24195

theorem fraction_ratio (x y : ℕ) (h : (x / y : ℚ) / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 27 ∧ y = 35 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_ratio_l241_24195


namespace NUMINAMATH_GPT_problem_1_problem_2_l241_24126

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | abs (x - 1) < a}

-- Define the first problem statement: If A ⊂ B, then a > 2.
theorem problem_1 (a : ℝ) : (A ⊂ B a) → (2 < a) := by
  sorry

-- Define the second problem statement: If B ⊂ A, then a ≤ 0 or (0 < a < 2).
theorem problem_2 (a : ℝ) : (B a ⊂ A) → (a ≤ 0 ∨ (0 < a ∧ a < 2)) := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l241_24126


namespace NUMINAMATH_GPT_find_a_value_l241_24100

theorem find_a_value
    (a : ℝ)
    (line : ∀ (x y : ℝ), 3 * x + y + a = 0)
    (circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0) :
    a = 1 := sorry

end NUMINAMATH_GPT_find_a_value_l241_24100


namespace NUMINAMATH_GPT_minimum_reflection_number_l241_24143

theorem minimum_reflection_number (a b : ℕ) :
  ((a + 2) * (b + 2) = 4042) ∧ (Nat.gcd (a + 1) (b + 1) = 1) → 
  (a + b = 129) :=
sorry

end NUMINAMATH_GPT_minimum_reflection_number_l241_24143


namespace NUMINAMATH_GPT_factorize_ax_squared_minus_9a_l241_24158

theorem factorize_ax_squared_minus_9a (a x : ℝ) : 
  a * x^2 - 9 * a = a * (x - 3) * (x + 3) :=
sorry

end NUMINAMATH_GPT_factorize_ax_squared_minus_9a_l241_24158


namespace NUMINAMATH_GPT_judy_expense_correct_l241_24142

noncomputable def judy_expense : ℝ :=
  let carrots := 5 * 1
  let milk := 3 * 3
  let pineapples := 2 * 4
  let original_flour_price := 5
  let discount := original_flour_price * 0.25
  let discounted_flour_price := original_flour_price - discount
  let flour := 2 * discounted_flour_price
  let ice_cream := 7
  let total_no_coupon := carrots + milk + pineapples + flour + ice_cream
  if total_no_coupon >= 30 then total_no_coupon - 10 else total_no_coupon

theorem judy_expense_correct : judy_expense = 26.5 := by
  sorry

end NUMINAMATH_GPT_judy_expense_correct_l241_24142


namespace NUMINAMATH_GPT_find_a_l241_24157

theorem find_a (a : ℝ) :
  (∀ x : ℝ, deriv (fun x => a * x^3 - 2) x * x = 1) → a = 1 / 3 :=
by
  intro h
  have slope_at_minus_1 := h (-1)
  sorry -- here we stop as proof isn't needed

end NUMINAMATH_GPT_find_a_l241_24157


namespace NUMINAMATH_GPT_overall_profit_percentage_l241_24191

theorem overall_profit_percentage :
  let SP_A := 900
  let SP_B := 1200
  let SP_C := 1500
  let P_A := 300
  let P_B := 400
  let P_C := 500
  let CP_A := SP_A - P_A
  let CP_B := SP_B - P_B
  let CP_C := SP_C - P_C
  let TCP := CP_A + CP_B + CP_C
  let TSP := SP_A + SP_B + SP_C
  let TP := TSP - TCP
  let ProfitPercentage := (TP / TCP) * 100
  ProfitPercentage = 50 := by
  sorry

end NUMINAMATH_GPT_overall_profit_percentage_l241_24191


namespace NUMINAMATH_GPT_sum_of_squares_inequality_l241_24122

theorem sum_of_squares_inequality (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_inequality_l241_24122


namespace NUMINAMATH_GPT_original_sandbox_capacity_l241_24149

theorem original_sandbox_capacity :
  ∃ (L W H : ℝ), 8 * (L * W * H) = 80 → L * W * H = 10 :=
by
  sorry

end NUMINAMATH_GPT_original_sandbox_capacity_l241_24149


namespace NUMINAMATH_GPT_factor_roots_l241_24127

noncomputable def checkRoots (a b c t : ℚ) : Prop :=
  a * t^2 + b * t + c = 0

theorem factor_roots (t : ℚ) :
  checkRoots 8 17 (-10) t ↔ t = 5/8 ∨ t = -2 := by
sorry

end NUMINAMATH_GPT_factor_roots_l241_24127


namespace NUMINAMATH_GPT_prove_op_eq_l241_24180

-- Define the new operation ⊕
def op (x y : ℝ) := x^3 - 2*y + x

-- State that for any k, k ⊕ (k ⊕ k) = -k^3 + 3k
theorem prove_op_eq (k : ℝ) : op k (op k k) = -k^3 + 3*k :=
by 
  sorry

end NUMINAMATH_GPT_prove_op_eq_l241_24180


namespace NUMINAMATH_GPT_distinct_numbers_in_T_l241_24134

-- Definitions of sequences as functions
def seq1 (k: ℕ) : ℕ := 5 * k - 3
def seq2 (l: ℕ) : ℕ := 8 * l - 5

-- Definition of sets A and B
def A : Finset ℕ := Finset.image seq1 (Finset.range 3000)
def B : Finset ℕ := Finset.image seq2 (Finset.range 3000)

-- Definition of set T as the union of A and B
def T := A ∪ B

-- Proof statement
theorem distinct_numbers_in_T : T.card = 5400 := by
  sorry

end NUMINAMATH_GPT_distinct_numbers_in_T_l241_24134


namespace NUMINAMATH_GPT_sequence_a_n_l241_24163

theorem sequence_a_n (a : ℕ → ℕ) (h₁ : a 1 = 1)
(h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = a (n / 2) + a ((n + 1) / 2)) :
∀ n : ℕ, a n = n :=
by
  -- skip the proof with sorry
  sorry

end NUMINAMATH_GPT_sequence_a_n_l241_24163


namespace NUMINAMATH_GPT_balls_in_boxes_l241_24113

theorem balls_in_boxes : 
  (number_of_ways : ℕ) = 52 :=
by
  let number_of_balls := 5
  let number_of_boxes := 4
  let balls_indistinguishable := true
  let boxes_distinguishable := true
  let max_balls_per_box := 3
  
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l241_24113


namespace NUMINAMATH_GPT_smallest_k_l241_24110

theorem smallest_k (n k : ℕ) (h1: 2000 < n) (h2: n < 3000)
  (h3: ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1) :
  k = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l241_24110


namespace NUMINAMATH_GPT_kris_fraction_l241_24192

-- Definitions based on problem conditions
def Trey (kris : ℕ) := 7 * kris
def Kristen := 12
def Trey_kristen_diff := 9
def Kris_fraction_to_Kristen (kris : ℕ) : ℚ := kris / Kristen

-- Theorem statement: Proving the required fraction
theorem kris_fraction (kris : ℕ) (h1 : Trey kris = Kristen + Trey_kristen_diff) : 
  Kris_fraction_to_Kristen kris = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_kris_fraction_l241_24192


namespace NUMINAMATH_GPT_jake_car_washes_l241_24189

theorem jake_car_washes :
  ∀ (washes_per_bottle cost_per_bottle total_spent weekly_washes : ℕ),
  washes_per_bottle = 4 →
  cost_per_bottle = 4 →
  total_spent = 20 →
  weekly_washes = 1 →
  (total_spent / cost_per_bottle) * washes_per_bottle / weekly_washes = 20 :=
by
  intros washes_per_bottle cost_per_bottle total_spent weekly_washes
  sorry

end NUMINAMATH_GPT_jake_car_washes_l241_24189


namespace NUMINAMATH_GPT_number_of_books_in_box_l241_24151

theorem number_of_books_in_box :
  ∀ (total_weight : ℕ) (empty_box_weight : ℕ) (book_weight : ℕ),
  total_weight = 42 →
  empty_box_weight = 6 →
  book_weight = 3 →
  (total_weight - empty_box_weight) / book_weight = 12 :=
by
  intros total_weight empty_box_weight book_weight htwe hebe hbw
  sorry

end NUMINAMATH_GPT_number_of_books_in_box_l241_24151


namespace NUMINAMATH_GPT_theater_ticket_cost_l241_24145

theorem theater_ticket_cost
  (num_persons : ℕ) 
  (num_children : ℕ) 
  (num_adults : ℕ)
  (children_ticket_cost : ℕ)
  (total_receipts_cents : ℕ)
  (A : ℕ) :
  num_persons = 280 →
  num_children = 80 →
  children_ticket_cost = 25 →
  total_receipts_cents = 14000 →
  num_adults = num_persons - num_children →
  200 * A + (num_children * children_ticket_cost) = total_receipts_cents →
  A = 60 :=
by
  intros h_num_persons h_num_children h_children_ticket_cost h_total_receipts_cents h_num_adults h_eqn
  sorry

end NUMINAMATH_GPT_theater_ticket_cost_l241_24145


namespace NUMINAMATH_GPT_lamp_post_ratio_l241_24170

theorem lamp_post_ratio (x k m : ℕ) (h1 : 9 * x = k) (h2 : 99 * x = m) : m = 11 * k :=
by sorry

end NUMINAMATH_GPT_lamp_post_ratio_l241_24170


namespace NUMINAMATH_GPT_integral_f_x_l241_24106

theorem integral_f_x (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ t in (0 : ℝ)..1, f t) : 
  ∫ t in (0 : ℝ)..1, f t = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_integral_f_x_l241_24106


namespace NUMINAMATH_GPT_worked_days_proof_l241_24129

theorem worked_days_proof (W N : ℕ) (hN : N = 24) (h0 : 100 * W = 25 * N) : W + N = 30 :=
by
  sorry

end NUMINAMATH_GPT_worked_days_proof_l241_24129


namespace NUMINAMATH_GPT_find_digit_A_l241_24120

theorem find_digit_A :
  ∃ A : ℕ, 
    2 * 10^6 + A * 10^5 + 9 * 10^4 + 9 * 10^3 + 5 * 10^2 + 6 * 10^1 + 1 = (3 * (523 + A)) ^ 2 
    ∧ A = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_digit_A_l241_24120


namespace NUMINAMATH_GPT_determine_q_l241_24186

theorem determine_q (q : ℕ) (h : 81^10 = 3^q) : q = 40 :=
by
  sorry

end NUMINAMATH_GPT_determine_q_l241_24186


namespace NUMINAMATH_GPT_shortest_side_length_triangle_l241_24150

noncomputable def triangle_min_angle_side_length (A B : ℝ) (c : ℝ) (tanA tanB : ℝ) (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) : ℝ :=
   Real.sqrt 2

theorem shortest_side_length_triangle {A B c : ℝ} {tanA tanB : ℝ} 
  (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) :
  triangle_min_angle_side_length A B c tanA tanB ha hb hc = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_shortest_side_length_triangle_l241_24150


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l241_24144

theorem quadratic_inequality_solution (a b: ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) ∧
  (2 + 3 = -a) ∧
  (2 * 3 = b) →
  ∀ x : ℝ, (b * x^2 + a * x + 1 > 0) ↔ (x < 1/3 ∨ x > 1/2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l241_24144


namespace NUMINAMATH_GPT_sin_squared_minus_cos_squared_l241_24152

theorem sin_squared_minus_cos_squared {α : ℝ} (h : Real.sin α = Real.sqrt 5 / 5) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_sin_squared_minus_cos_squared_l241_24152


namespace NUMINAMATH_GPT_fraction_div_addition_l241_24112

theorem fraction_div_addition : ( (3 / 7 : ℚ) / 4) + (1 / 28) = (1 / 7) :=
  sorry

end NUMINAMATH_GPT_fraction_div_addition_l241_24112


namespace NUMINAMATH_GPT_problem_statement_l241_24190

-- Define the function
def f (x : ℝ) := -2 * x^2

-- We need to show that f is monotonically decreasing and even on (0, +∞)
theorem problem_statement : (∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x) ∧ (∀ x : ℝ, f (-x) = f x) := 
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_problem_statement_l241_24190


namespace NUMINAMATH_GPT_value_of_s_l241_24141

theorem value_of_s (s : ℝ) : (3 * (-1)^5 + 2 * (-1)^4 - (-1)^3 + (-1)^2 - 4 * (-1) + s = 0) → (s = -5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_s_l241_24141


namespace NUMINAMATH_GPT_max_value_of_expr_l241_24147

theorem max_value_of_expr (x : ℝ) (h : x ≠ 0) : 
  (∀ y : ℝ, y = (x^2) / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) → y ≤ 1/8) :=
sorry

end NUMINAMATH_GPT_max_value_of_expr_l241_24147


namespace NUMINAMATH_GPT_farmer_steven_total_days_l241_24116

theorem farmer_steven_total_days 
(plow_acres_per_day : ℕ)
(mow_acres_per_day : ℕ)
(farmland_acres : ℕ)
(grassland_acres : ℕ)
(h_plow : plow_acres_per_day = 10)
(h_mow : mow_acres_per_day = 12)
(h_farmland : farmland_acres = 55)
(h_grassland : grassland_acres = 30) :
((farmland_acres / plow_acres_per_day) + (grassland_acres / mow_acres_per_day) = 8) := by
  sorry

end NUMINAMATH_GPT_farmer_steven_total_days_l241_24116


namespace NUMINAMATH_GPT_original_height_of_tree_l241_24174

theorem original_height_of_tree
  (current_height_in_inches : ℕ)
  (percent_taller : ℕ)
  (current_height_is_V := 180)
  (percent_taller_is_50 := 50) :
  (current_height_in_inches * 100) / (percent_taller + 100) / 12 = 10 := sorry

end NUMINAMATH_GPT_original_height_of_tree_l241_24174


namespace NUMINAMATH_GPT_problem1_problem2_l241_24159

-- Problem (1)
variables {p q : ℝ}

theorem problem1 (hpq : p^3 + q^3 = 2) : p + q ≤ 2 := sorry

-- Problem (2)
variables {a b : ℝ}

theorem problem2 (hab : |a| + |b| < 1) : ∀ x : ℝ, (x^2 + a * x + b = 0) → |x| < 1 := sorry

end NUMINAMATH_GPT_problem1_problem2_l241_24159


namespace NUMINAMATH_GPT_three_wheels_possible_two_wheels_not_possible_l241_24103

-- Define the conditions as hypotheses
def wheels_spokes (total_spokes_visible : ℕ) (max_spokes_per_wheel : ℕ) (wheels : ℕ) : Prop :=
  total_spokes_visible >= wheels * max_spokes_per_wheel ∧ wheels ≥ 1

-- Prove if a) three wheels is a possible solution
theorem three_wheels_possible : ∃ wheels, wheels = 3 ∧ wheels_spokes 7 3 wheels := by
  sorry

-- Prove if b) two wheels is not a possible solution
theorem two_wheels_not_possible : ¬ ∃ wheels, wheels = 2 ∧ wheels_spokes 7 3 wheels := by
  sorry

end NUMINAMATH_GPT_three_wheels_possible_two_wheels_not_possible_l241_24103


namespace NUMINAMATH_GPT_height_at_end_of_2_years_l241_24140

-- Step d): Define the conditions and state the theorem

-- Define a function modeling the height of the tree each year
def tree_height (initial_height : ℕ) (years : ℕ) : ℕ :=
  initial_height * 3^years

-- Given conditions as definitions
def year_4_height := 81 -- height at the end of 4 years

-- Theorem that we need to prove
theorem height_at_end_of_2_years (initial_height : ℕ) (h : tree_height initial_height 4 = year_4_height) :
  tree_height initial_height 2 = 9 :=
sorry

end NUMINAMATH_GPT_height_at_end_of_2_years_l241_24140


namespace NUMINAMATH_GPT_campers_difference_l241_24165

theorem campers_difference (a_morning : ℕ) (b_morning_afternoon : ℕ) (a_afternoon : ℕ) (a_afternoon_evening : ℕ) (c_evening_only : ℕ) :
  a_morning = 33 ∧ b_morning_afternoon = 11 ∧ a_afternoon = 34 ∧ a_afternoon_evening = 20 ∧ c_evening_only = 10 →
  a_afternoon - (a_afternoon_evening + c_evening_only) = 4 := 
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_campers_difference_l241_24165


namespace NUMINAMATH_GPT_quadratic_one_root_iff_discriminant_zero_l241_24169

theorem quadratic_one_root_iff_discriminant_zero (m : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y^2 - m*y + 1 ≤ 0 ↔ y = x) ↔ (m = 2 ∨ m = -2) :=
by 
  -- We assume the discriminant condition which implies the result
  sorry

end NUMINAMATH_GPT_quadratic_one_root_iff_discriminant_zero_l241_24169


namespace NUMINAMATH_GPT_calculate_expression_l241_24117

theorem calculate_expression :
  (-3)^4 + (-3)^3 + 3^3 + 3^4 = 162 :=
by
  -- Since all necessary conditions are listed in the problem statement, we honor this structure
  -- The following steps are required logically but are not presently necessary for detailed proof means.
  sorry

end NUMINAMATH_GPT_calculate_expression_l241_24117


namespace NUMINAMATH_GPT_length_of_each_train_l241_24154

theorem length_of_each_train
  (L : ℝ) -- length of each train
  (speed_fast : ℝ) (speed_slow : ℝ) -- speeds of the fast and slow trains in km/hr
  (time_pass : ℝ) -- time for the slower train to pass the driver of the faster one in seconds
  (h_speed_fast : speed_fast = 45) -- speed of the faster train
  (h_speed_slow : speed_slow = 15) -- speed of the slower train
  (h_time_pass : time_pass = 60) -- time to pass
  (h_same_length : ∀ (x y : ℝ), x = y → x = L) :  
  L = 1000 :=
  by
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_length_of_each_train_l241_24154


namespace NUMINAMATH_GPT_ratio_of_bottles_given_to_first_house_l241_24161

theorem ratio_of_bottles_given_to_first_house 
  (total_bottles : ℕ) 
  (bottles_only_cider : ℕ) 
  (bottles_only_beer : ℕ) 
  (bottles_mixed : ℕ) 
  (first_house_bottles : ℕ) 
  (h1 : total_bottles = 180) 
  (h2 : bottles_only_cider = 40) 
  (h3 : bottles_only_beer = 80) 
  (h4 : bottles_mixed = total_bottles - bottles_only_cider - bottles_only_beer) 
  (h5 : first_house_bottles = 90) : 
  first_house_bottles / total_bottles = 1 / 2 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_bottles_given_to_first_house_l241_24161


namespace NUMINAMATH_GPT_find_value_of_b_l241_24153

theorem find_value_of_b (x b : ℕ) 
    (h1 : 5 * (x + 8) = 5 * x + b + 33) : b = 7 :=
sorry

end NUMINAMATH_GPT_find_value_of_b_l241_24153


namespace NUMINAMATH_GPT_parabola_bisects_rectangle_l241_24194
open Real

theorem parabola_bisects_rectangle (a : ℝ) (h_pos : a > 0) : 
  ((a^3 + a) / 2 = (a^3 / 3 + a)) → a = sqrt 3 := by
  sorry

end NUMINAMATH_GPT_parabola_bisects_rectangle_l241_24194


namespace NUMINAMATH_GPT_number_properties_l241_24114

def number : ℕ := 52300600

def position_of_2 : ℕ := 10^6

def value_of_2 : ℕ := 20000000

def position_of_5 : ℕ := 10^7

def value_of_5 : ℕ := 50000000

def read_number : String := "five hundred twenty-three million six hundred"

theorem number_properties : 
  position_of_2 = (10^6) ∧ value_of_2 = 20000000 ∧ 
  position_of_5 = (10^7) ∧ value_of_5 = 50000000 ∧ 
  read_number = "five hundred twenty-three million six hundred" :=
by sorry

end NUMINAMATH_GPT_number_properties_l241_24114


namespace NUMINAMATH_GPT_area_of_given_sector_l241_24177

noncomputable def area_of_sector (alpha l : ℝ) : ℝ :=
  let r := l / alpha
  (1 / 2) * l * r

theorem area_of_given_sector :
  let alpha := Real.pi / 9
  let l := Real.pi / 3
  area_of_sector alpha l = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_given_sector_l241_24177


namespace NUMINAMATH_GPT_problem_solution_l241_24178

theorem problem_solution (x y z : ℝ) (h1 : 2 * x - y - 2 * z - 6 = 0) (h2 : x^2 + y^2 + z^2 ≤ 4) :
  2 * x + y + z = 2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l241_24178


namespace NUMINAMATH_GPT_boston_trip_distance_l241_24123

theorem boston_trip_distance :
  ∃ d : ℕ, 40 * d = 440 :=
by
  sorry

end NUMINAMATH_GPT_boston_trip_distance_l241_24123


namespace NUMINAMATH_GPT_nth_permutation_2013_eq_3546127_l241_24121

-- Given the digits 1 through 7, there are 7! = 5040 permutations.
-- We want to prove that the 2013th permutation in ascending order is 3546127.

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def nth_permutation (n : ℕ) (digits : List ℕ) : List ℕ :=
  sorry

theorem nth_permutation_2013_eq_3546127 :
  nth_permutation 2013 digits = [3, 5, 4, 6, 1, 2, 7] :=
sorry

end NUMINAMATH_GPT_nth_permutation_2013_eq_3546127_l241_24121


namespace NUMINAMATH_GPT_find_k_eq_neg_four_thirds_l241_24104

-- Definitions based on conditions
def hash_p (k : ℚ) (p : ℚ) : ℚ := k * p + 20

-- Using the initial condition
def triple_hash_18 (k : ℚ) : ℚ :=
  let hp := hash_p k 18
  let hhp := hash_p k hp
  hash_p k hhp

-- The Lean statement for the desired proof
theorem find_k_eq_neg_four_thirds (k : ℚ) (h : triple_hash_18 k = -4) : k = -4 / 3 :=
sorry

end NUMINAMATH_GPT_find_k_eq_neg_four_thirds_l241_24104


namespace NUMINAMATH_GPT_cube_volume_l241_24176

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end NUMINAMATH_GPT_cube_volume_l241_24176


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_eq_pow_of_two_l241_24109

theorem sum_of_consecutive_integers_eq_pow_of_two (n : ℕ) : 
  (∀ a b : ℕ, a < b → 2 * n ≠ (a + b) * (b - a + 1)) ↔ ∃ k : ℕ, n = 2 ^ k := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_eq_pow_of_two_l241_24109


namespace NUMINAMATH_GPT_Taimour_paint_time_l241_24156

theorem Taimour_paint_time (T : ℝ) (H1 : ∀ t : ℝ, t = 2 / T → t ≠ 0) (H2 : (1 / T + 2 / T) = 1 / 3) : T = 9 :=
by
  sorry

end NUMINAMATH_GPT_Taimour_paint_time_l241_24156


namespace NUMINAMATH_GPT_male_students_tree_planting_l241_24160

theorem male_students_tree_planting (average_trees : ℕ) (female_trees : ℕ) 
    (male_trees : ℕ) : 
    (average_trees = 6) →
    (female_trees = 15) → 
    (1 / male_trees + 1 / female_trees = 1 / average_trees) → 
    male_trees = 10 :=
by
  intros h_avg h_fem h_eq
  sorry

end NUMINAMATH_GPT_male_students_tree_planting_l241_24160


namespace NUMINAMATH_GPT_units_digit_7_pow_2050_l241_24111

theorem units_digit_7_pow_2050 : (7 ^ 2050) % 10 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_2050_l241_24111


namespace NUMINAMATH_GPT_complex_star_angle_sum_correct_l241_24136

-- Definitions corresponding to the conditions
def complex_star_interior_angle_sum (n : ℕ) (h : n ≥ 7) : ℕ :=
  180 * (n - 4)

-- The theorem stating the problem
theorem complex_star_angle_sum_correct (n : ℕ) (h : n ≥ 7) :
  complex_star_interior_angle_sum n h = 180 * (n - 4) :=
sorry

end NUMINAMATH_GPT_complex_star_angle_sum_correct_l241_24136


namespace NUMINAMATH_GPT_difference_value_l241_24167

theorem difference_value (N : ℝ) (h : 0.25 * N = 100) : N - (3/4) * N = 100 :=
by sorry

end NUMINAMATH_GPT_difference_value_l241_24167


namespace NUMINAMATH_GPT_sequence_a8_value_l241_24181

theorem sequence_a8_value :
  ∃ a : ℕ → ℚ, a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) / a n = n / (n + 1)) ∧ a 8 = 1 / 8 :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_sequence_a8_value_l241_24181


namespace NUMINAMATH_GPT_kelly_games_left_l241_24168

theorem kelly_games_left (initial_games : Nat) (given_away : Nat) (remaining_games : Nat) 
  (h1 : initial_games = 106) (h2 : given_away = 64) : remaining_games = 42 := by
  sorry

end NUMINAMATH_GPT_kelly_games_left_l241_24168


namespace NUMINAMATH_GPT_fg_of_3_eq_29_l241_24148

def g (x : ℝ) : ℝ := x^2
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_eq_29_l241_24148


namespace NUMINAMATH_GPT_area_RWP_l241_24197

-- Definitions
variables (X Y Z W P Q R : ℝ × ℝ)
variables (h₁ : (X.1 - Z.1) * (X.1 - Z.1) + (X.2 - Z.2) * (X.2 - Z.2) = 144)
variables (h₂ : P.1 = X.1 - 8 ∧ P.2 = X.2)
variables (h₃ : Q.1 = (Z.1 + P.1) / 2 ∧ Q.2 = (Z.2 + P.2) / 2)
variables (h₄ : R.1 = (Y.1 + P.1) / 2 ∧ R.2 = (Y.2 + P.2) / 2)
variables (h₅ : 1 / 2 * ((Z.1 - X.1) * (W.2 - X.2) - (Z.2 - X.2) * (W.1 - X.1)) = 72)
variables (h₆ : 1 / 2 * abs ((Q.1 - X.1) * (W.2 - X.2) - (Q.2 - X.2) * (W.1 - X.1)) = 20)

-- Theorem statement
theorem area_RWP : 
  1 / 2 * abs ((R.1 - W.1) * (P.2 - W.2) - (R.2 - W.2) * (P.1 - W.1)) = 12 :=
sorry

end NUMINAMATH_GPT_area_RWP_l241_24197


namespace NUMINAMATH_GPT_g_iterated_six_times_is_2_l241_24171

def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem g_iterated_six_times_is_2 : g (g (g (g (g (g 2))))) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_g_iterated_six_times_is_2_l241_24171


namespace NUMINAMATH_GPT_line_is_x_axis_l241_24130

theorem line_is_x_axis (A B C : ℝ) (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : A = 0 ∧ B ≠ 0 ∧ C = 0 :=
by sorry

end NUMINAMATH_GPT_line_is_x_axis_l241_24130


namespace NUMINAMATH_GPT_new_number_formed_l241_24124

theorem new_number_formed (t u : ℕ) (ht : t < 10) (hu : u < 10) : 3 * 100 + (10 * t + u) = 300 + 10 * t + u := 
by {
  sorry
}

end NUMINAMATH_GPT_new_number_formed_l241_24124


namespace NUMINAMATH_GPT_complex_props_hold_l241_24187

theorem complex_props_hold (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ (a^2 = a*b → a = b) :=
by
  sorry

end NUMINAMATH_GPT_complex_props_hold_l241_24187


namespace NUMINAMATH_GPT_addition_problem_l241_24108

theorem addition_problem (m n p q : ℕ) (Hm : m = 2) (Hn : 2 + n + 7 + 5 = 20) (Hp : 1 + 6 + p + 8 = 24) (Hq : 3 + 2 + q = 12) (Hpositives : 0 < m ∧ 0 < n ∧ 0 < p ∧ 0 < q) :
  m + n + p + q = 24 :=
sorry

end NUMINAMATH_GPT_addition_problem_l241_24108


namespace NUMINAMATH_GPT_cos_A_minus_cos_C_l241_24139

-- Definitions representing the conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
variables (h₂ : 2 * b = a + c) (h₃ : A < B) (h₄ : B < C)

-- Statement of the proof problem
theorem cos_A_minus_cos_C (A B C a b c : ℝ)
  (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h₂ : 2 * b = a + c)
  (h₃ : A < B)
  (h₄ : B < C) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_A_minus_cos_C_l241_24139


namespace NUMINAMATH_GPT_ratio_of_roots_l241_24199

theorem ratio_of_roots (c : ℝ) :
  (∃ (x1 x2 : ℝ), 5 * x1^2 - 2 * x1 + c = 0 ∧ 5 * x2^2 - 2 * x2 + c = 0 ∧ x1 / x2 = -3 / 5) → c = -3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_roots_l241_24199


namespace NUMINAMATH_GPT_power_function_evaluation_l241_24193

noncomputable def f (α : ℝ) (x : ℝ) := x ^ α

theorem power_function_evaluation (α : ℝ) (h : f α 8 = 2) : f α (-1/8) = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_power_function_evaluation_l241_24193


namespace NUMINAMATH_GPT_eval_expression_correct_l241_24173

noncomputable def evaluate_expression : ℝ :=
    3 + Real.sqrt 3 + (3 - Real.sqrt 3) / 6 + (1 / (Real.cos (Real.pi / 4) - 3))

theorem eval_expression_correct : 
  evaluate_expression = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 :=
by
  -- Proof can be filled in later
  sorry

end NUMINAMATH_GPT_eval_expression_correct_l241_24173
