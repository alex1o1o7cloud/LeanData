import Mathlib

namespace jean_spots_on_sides_l611_61147

variables (total_spots upper_torso_spots back_hindquarters_spots side_spots : ℕ)

def half (x : ℕ) := x / 2
def third (x : ℕ) := x / 3

-- Given conditions
axiom h1 : upper_torso_spots = 30
axiom h2 : upper_torso_spots = half total_spots
axiom h3 : back_hindquarters_spots = third total_spots
axiom h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

-- Theorem to prove
theorem jean_spots_on_sides (h1 : upper_torso_spots = 30)
    (h2 : upper_torso_spots = half total_spots)
    (h3 : back_hindquarters_spots = third total_spots)
    (h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots) :
    side_spots = 10 := by
  sorry

end jean_spots_on_sides_l611_61147


namespace t_shirts_left_yesterday_correct_l611_61194

-- Define the conditions
def t_shirts_left_yesterday (x : ℕ) : Prop :=
  let t_shirts_sold_morning := (3 / 5) * x
  let t_shirts_sold_afternoon := 180
  t_shirts_sold_morning = t_shirts_sold_afternoon

-- Prove that x = 300 given the above conditions
theorem t_shirts_left_yesterday_correct (x : ℕ) (h : t_shirts_left_yesterday x) : x = 300 :=
by
  sorry

end t_shirts_left_yesterday_correct_l611_61194


namespace complex_division_correct_l611_61117

theorem complex_division_correct : (3 - 1 * Complex.I) / (1 + Complex.I) = 1 - 2 * Complex.I := 
by
  sorry

end complex_division_correct_l611_61117


namespace prove_RoseHasMoney_l611_61138
noncomputable def RoseHasMoney : Prop :=
  let cost_of_paintbrush := 2.40
  let cost_of_paints := 9.20
  let cost_of_easel := 6.50
  let total_cost := cost_of_paintbrush + cost_of_paints + cost_of_easel
  let additional_money_needed := 11
  let money_rose_has := total_cost - additional_money_needed
  money_rose_has = 7.10

theorem prove_RoseHasMoney : RoseHasMoney :=
  sorry

end prove_RoseHasMoney_l611_61138


namespace max_items_for_2019_students_l611_61101

noncomputable def max_items (students : ℕ) : ℕ :=
  students / 2

theorem max_items_for_2019_students : max_items 2019 = 1009 := by
  sorry

end max_items_for_2019_students_l611_61101


namespace sum_possible_distances_l611_61109

theorem sum_possible_distances {A B : ℝ} (hAB : |A - B| = 2) (hA : |A| = 3) : 
  (if A = 3 then |B + 2| + |B - 2| else |B + 4| + |B - 4|) = 12 :=
by
  sorry

end sum_possible_distances_l611_61109


namespace range_of_m_l611_61132

noncomputable def f : ℝ → ℝ := sorry

lemma function_symmetric {x : ℝ} : f (2 + x) = f (-x) := sorry

lemma f_decreasing_on_pos_halfline {x y : ℝ} (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) : f x ≥ f y := sorry

theorem range_of_m {m : ℝ} (h : f (1 - m) < f m) : m > (1 / 2) := sorry

end range_of_m_l611_61132


namespace find_views_multiplier_l611_61179

theorem find_views_multiplier (M: ℝ) (h: 4000 * M + 50000 = 94000) : M = 11 :=
by
  sorry

end find_views_multiplier_l611_61179


namespace spiders_make_webs_l611_61196

theorem spiders_make_webs :
  (∀ (s d : ℕ), s = 7 ∧ d = 7 → (∃ w : ℕ, w = s)) ∧
  (∀ (d w : ℕ), w = 1 ∧ d = 7 → (∃ s : ℕ, s = w)) →
  (∀ (s : ℕ), s = 1) :=
by
  sorry

end spiders_make_webs_l611_61196


namespace fraction_of_sum_after_6_years_l611_61140

-- Define the principal amount, rate, and time period as given in the conditions
def P : ℝ := 1
def R : ℝ := 0.02777777777777779
def T : ℕ := 6

-- Definition of the Simple Interest calculation
def simple_interest (P R : ℝ) (T : ℕ) : ℝ :=
  P * R * T

-- Definition of the total amount after 6 years
def total_amount (P SI : ℝ) : ℝ :=
  P + SI

-- The main theorem to prove
theorem fraction_of_sum_after_6_years :
  total_amount P (simple_interest P R T) = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_after_6_years_l611_61140


namespace total_people_surveyed_l611_61123

theorem total_people_surveyed (x y : ℝ) (h1 : 0.536 * x = 30) (h2 : 0.794 * y = x) : y = 71 :=
by
  sorry

end total_people_surveyed_l611_61123


namespace intersection_point_l611_61163

theorem intersection_point :
  ∃ (x y : ℝ), (2 * x + 3 * y + 8 = 0) ∧ (x - y - 1 = 0) ∧ (x = -1) ∧ (y = -2) := 
by
  sorry

end intersection_point_l611_61163


namespace equilateral_triangle_iff_l611_61168

theorem equilateral_triangle_iff (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c :=
sorry

end equilateral_triangle_iff_l611_61168


namespace camden_dogs_fraction_l611_61197

def number_of_dogs (Justins_dogs : ℕ) (extra_dogs : ℕ) : ℕ := Justins_dogs + extra_dogs
def dogs_from_legs (total_legs : ℕ) (legs_per_dog : ℕ) : ℕ := total_legs / legs_per_dog
def fraction_of_dogs (dogs_camden : ℕ) (dogs_rico : ℕ) : ℚ := dogs_camden / dogs_rico

theorem camden_dogs_fraction (Justins_dogs : ℕ) (extra_dogs : ℕ) (total_legs_camden : ℕ) (legs_per_dog : ℕ) :
  Justins_dogs = 14 →
  extra_dogs = 10 →
  total_legs_camden = 72 →
  legs_per_dog = 4 →
  fraction_of_dogs (dogs_from_legs total_legs_camden legs_per_dog) (number_of_dogs Justins_dogs extra_dogs) = 3 / 4 :=
by
  sorry

end camden_dogs_fraction_l611_61197


namespace shoe_size_ratio_l611_61143

theorem shoe_size_ratio (J A : ℕ) (hJ : J = 7) (hAJ : A + J = 21) : A / J = 2 :=
by
  -- Skipping the proof
  sorry

end shoe_size_ratio_l611_61143


namespace remainder_13_plus_y_l611_61112

theorem remainder_13_plus_y :
  (∃ y : ℕ, (0 < y) ∧ (7 * y ≡ 1 [MOD 31])) → (∃ y : ℕ, (13 + y ≡ 22 [MOD 31])) :=
by 
  sorry

end remainder_13_plus_y_l611_61112


namespace expected_value_of_groups_l611_61100

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end expected_value_of_groups_l611_61100


namespace samantha_coins_value_l611_61129

theorem samantha_coins_value (n d : ℕ) (h1 : n + d = 25) 
    (original_value : ℕ := 250 - 5 * n) 
    (swapped_value : ℕ := 125 + 5 * n)
    (h2 : swapped_value = original_value + 100) : original_value = 140 := 
by
  sorry

end samantha_coins_value_l611_61129


namespace parametric_equations_curveC2_minimum_distance_M_to_curveC_l611_61173

noncomputable def curveC1_param (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α)

def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem parametric_equations_curveC2 (θ : ℝ) :
  scaling_transform (Real.cos θ) (Real.sin θ) = (3 * Real.cos θ, 2 * Real.sin θ) :=
sorry

noncomputable def curveC (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin θ + ρ * Real.cos θ = 10

noncomputable def distance_to_curveC (θ : ℝ) : ℝ :=
  abs (3 * Real.cos θ + 4 * Real.sin θ - 10) / Real.sqrt 5

theorem minimum_distance_M_to_curveC : 
  ∀ θ, distance_to_curveC θ >= Real.sqrt 5 :=
sorry

end parametric_equations_curveC2_minimum_distance_M_to_curveC_l611_61173


namespace original_numbers_l611_61114

theorem original_numbers (a b c d : ℕ) (x : ℕ)
  (h1 : a + b + c + d = 45)
  (h2 : a + 2 = x)
  (h3 : b - 2 = x)
  (h4 : 2 * c = x)
  (h5 : d / 2 = x) : 
  (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
sorry

end original_numbers_l611_61114


namespace line_equation_l611_61146

/-
Given points M(2, 3) and N(4, -5), and a line l passes through the 
point P(1, 2). Prove that the line l has equal distances from points 
M and N if and only if its equation is either 4x + y - 6 = 0 or 
3x + 2y - 7 = 0.
-/

theorem line_equation (M N P : ℝ × ℝ)
(hM : M = (2, 3))
(hN : N = (4, -5))
(hP : P = (1, 2))
(l : ℝ → ℝ → Prop)
(h_l : ∀ x y, l x y ↔ (4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0))
: ∀ (dM dN : ℝ), 
(∀ x y , l x y → (x = 1) → (y = 2) ∧ (|M.1 - x| + |M.2 - y| = |N.1 - x| + |N.2 - y|)) :=
sorry

end line_equation_l611_61146


namespace downstream_speed_l611_61182

def Vm : ℝ := 31  -- speed in still water
def Vu : ℝ := 25  -- speed upstream
def Vs := Vm - Vu  -- speed of stream

theorem downstream_speed : Vm + Vs = 37 := 
by
  sorry

end downstream_speed_l611_61182


namespace bread_cost_l611_61151

theorem bread_cost
  (B : ℝ)
  (cost_peanut_butter : ℝ := 2)
  (initial_money : ℝ := 14)
  (money_leftover : ℝ := 5.25) :
  3 * B + cost_peanut_butter = (initial_money - money_leftover) → B = 2.25 :=
by
  sorry

end bread_cost_l611_61151


namespace find_XY_sum_in_base10_l611_61156

def base8_addition_step1 (X : ℕ) : Prop :=
  X + 5 = 9

def base8_addition_step2 (Y X : ℕ) : Prop :=
  Y + 3 = X

theorem find_XY_sum_in_base10 (X Y : ℕ) (h1 : base8_addition_step1 X) (h2 : base8_addition_step2 Y X) :
  X + Y = 5 :=
by
  sorry

end find_XY_sum_in_base10_l611_61156


namespace find_a12_a12_value_l611_61187

variable (a : ℕ → ℝ)

-- Given conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

axiom h1 : a 6 + a 10 = 16
axiom h2 : a 4 = 1

-- Theorem to prove
theorem find_a12 : a 6 + a 10 = a 4 + a 12 := by
  -- Place for the proof
  sorry

theorem a12_value : (∃ a12, a 6 + a 10 = 16 ∧ a 4 = 1 ∧ a 6 + a 10 = a 4 + a12) → a 12 = 15 :=
by
  -- Place for the proof
  sorry

end find_a12_a12_value_l611_61187


namespace color_of_85th_bead_l611_61189

/-- Definition for the repeating pattern of beads -/
def pattern : List String := ["red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

/-- Definition for finding the color of the n-th bead -/
def bead_color (n : Nat) : Option String :=
  let index := (n - 1) % pattern.length
  pattern.get? index

theorem color_of_85th_bead : bead_color 85 = some "yellow" := by
  sorry

end color_of_85th_bead_l611_61189


namespace total_worth_of_stock_l611_61130

theorem total_worth_of_stock :
  let cost_expensive := 10
  let cost_cheaper := 3.5
  let total_modules := 11
  let cheaper_modules := 10
  let expensive_modules := total_modules - cheaper_modules
  let worth_cheaper_modules := cheaper_modules * cost_cheaper
  let worth_expensive_module := expensive_modules * cost_expensive 
  worth_cheaper_modules + worth_expensive_module = 45 := by
  sorry

end total_worth_of_stock_l611_61130


namespace jared_sent_in_november_l611_61145

noncomputable def text_messages (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- November
  | 1 => 2  -- December
  | 2 => 4  -- January
  | 3 => 8  -- February
  | 4 => 16 -- March
  | _ => 0

theorem jared_sent_in_november : text_messages 0 = 1 :=
sorry

end jared_sent_in_november_l611_61145


namespace most_suitable_sampling_method_l611_61152

/-- A unit has 28 elderly people, 54 middle-aged people, and 81 young people. 
    A sample of 36 people needs to be drawn in a way that accounts for age.
    The most suitable method for drawing a sample is to exclude one elderly person first,
    then use stratified sampling. -/
theorem most_suitable_sampling_method 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (sample_size : ℕ) (suitable_method : String)
  (condition1 : elderly = 28) 
  (condition2 : middle_aged = 54) 
  (condition3 : young = 81) 
  (condition4 : sample_size = 36) 
  (condition5 : suitable_method = "Exclude one elderly person first, then stratify sampling") : 
  suitable_method = "Exclude one elderly person first, then stratify sampling" := 
by sorry

end most_suitable_sampling_method_l611_61152


namespace min_tiles_needed_l611_61183

theorem min_tiles_needed : 
  ∀ (tile_length tile_width region_length region_width: ℕ),
  tile_length = 5 → 
  tile_width = 6 → 
  region_length = 3 * 12 → 
  region_width = 4 * 12 → 
  (region_length * region_width) / (tile_length * tile_width) ≤ 58 :=
by
  intros tile_length tile_width region_length region_width h_tile_length h_tile_width h_region_length h_region_width
  sorry

end min_tiles_needed_l611_61183


namespace count_integers_l611_61161

def Q (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 9) * (x - 16) * (x - 25) * (x - 36) * (x - 49) * (x - 64) * (x - 81)

theorem count_integers (Q_le_0 : ∀ n : ℤ, Q n ≤ 0 → ∃ k : ℕ, k = 53) : ∃ k : ℕ, k = 53 := by
  sorry

end count_integers_l611_61161


namespace determine_values_a_b_l611_61108

theorem determine_values_a_b (a b x : ℝ) (h₁ : x > 1)
  (h₂ : 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = (10 * (Real.log x)^2) / (Real.log a + Real.log b)) :
  b = a ^ ((5 + Real.sqrt 10) / 3) ∨ b = a ^ ((5 - Real.sqrt 10) / 3) :=
by sorry

end determine_values_a_b_l611_61108


namespace alex_initial_silk_l611_61170

theorem alex_initial_silk (m_per_dress : ℕ) (m_per_friend : ℕ) (num_friends : ℕ) (num_dresses : ℕ) (initial_silk : ℕ) :
  m_per_dress = 5 ∧ m_per_friend = 20 ∧ num_friends = 5 ∧ num_dresses = 100 ∧ 
  (initial_silk - (num_friends * m_per_friend)) / m_per_dress * m_per_dress = num_dresses * m_per_dress → 
  initial_silk = 600 :=
by
  intros
  sorry

end alex_initial_silk_l611_61170


namespace positive_triple_l611_61116

theorem positive_triple
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : ab + bc + ca > 0)
  (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_triple_l611_61116


namespace average_of_remaining_two_numbers_l611_61118

theorem average_of_remaining_two_numbers (S a₁ a₂ a₃ a₄ : ℝ)
    (h₁ : S / 6 = 3.95)
    (h₂ : (a₁ + a₂) / 2 = 3.8)
    (h₃ : (a₃ + a₄) / 2 = 3.85) :
    (S - (a₁ + a₂ + a₃ + a₄)) / 2 = 4.2 := 
sorry

end average_of_remaining_two_numbers_l611_61118


namespace salary_of_A_l611_61139

-- Given:
-- A + B = 6000
-- A's savings = 0.05A
-- B's savings = 0.15B
-- A's savings = B's savings

theorem salary_of_A (A B : ℝ) (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) :
  A = 4500 :=
sorry

end salary_of_A_l611_61139


namespace gift_distribution_l611_61115

theorem gift_distribution :
  let bags := [1, 2, 3, 4, 5]
  let num_people := 4
  ∃ d: ℕ, d = 96 := by
  -- Proof to be completed
  sorry

end gift_distribution_l611_61115


namespace part1_part2_l611_61181

def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 2| + x
def g (x : ℝ) : ℝ := |x - 2| - |2 * x - 3| + x

theorem part1 (a : ℝ) : (∀ x, f a x ≤ f a 2) ↔ a ≤ -1 :=
by sorry

theorem part2 (x : ℝ) : f 1 x < |2 * x - 3| ↔ x > 0.5 :=
by sorry

end part1_part2_l611_61181


namespace speed_in_still_water_l611_61153

theorem speed_in_still_water (U D : ℝ) (hU : U = 15) (hD : D = 25) : (U + D) / 2 = 20 :=
by
  rw [hU, hD]
  norm_num

end speed_in_still_water_l611_61153


namespace gcd_exponential_identity_l611_61162

theorem gcd_exponential_identity (a b : ℕ) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := sorry

end gcd_exponential_identity_l611_61162


namespace solution_range_of_a_l611_61157

theorem solution_range_of_a (a : ℝ) (x y : ℝ) :
  3 * x + y = 1 + a → x + 3 * y = 3 → x + y < 2 → a < 4 :=
by
  sorry

end solution_range_of_a_l611_61157


namespace inequality_solution_l611_61103

theorem inequality_solution (x : ℝ) :
  (-1 : ℝ) < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < (1 : ℝ) ↔
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by
  sorry

end inequality_solution_l611_61103


namespace solve_for_a_l611_61191

theorem solve_for_a (a : ℕ) (h : a > 0) (eqn : a / (a + 37) = 925 / 1000) : a = 455 :=
sorry

end solve_for_a_l611_61191


namespace cape_may_shark_sightings_l611_61104

def total_shark_sightings (D C : ℕ) : Prop :=
  D + C = 40

def cape_may_sightings (D C : ℕ) : Prop :=
  C = 2 * D - 8

theorem cape_may_shark_sightings : 
  ∃ (C D : ℕ), total_shark_sightings D C ∧ cape_may_sightings D C ∧ C = 24 :=
by
  sorry

end cape_may_shark_sightings_l611_61104


namespace max_value_sqrt_expression_l611_61107

noncomputable def expression_max_value (a b: ℝ) : ℝ :=
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b))

theorem max_value_sqrt_expression : 
  ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → expression_max_value a b ≤ 1 :=
by
  intros a b h
  sorry

end max_value_sqrt_expression_l611_61107


namespace quarters_and_dimes_l611_61121

theorem quarters_and_dimes (n : ℕ) (qval : ℕ := 25) (dval : ℕ := 10) 
  (hq : 20 * qval + 10 * dval = 10 * qval + n * dval) : 
  n = 35 :=
by
  sorry

end quarters_and_dimes_l611_61121


namespace ages_correct_l611_61120

-- Let A be Anya's age and P be Petya's age
def anya_age : ℕ := 4
def petya_age : ℕ := 12

-- The conditions
def condition1 (A P : ℕ) : Prop := P = 3 * A
def condition2 (A P : ℕ) : Prop := P - A = 8

-- The statement to be proven
theorem ages_correct : condition1 anya_age petya_age ∧ condition2 anya_age petya_age :=
by
  unfold condition1 condition2 anya_age petya_age -- Reveal the definitions
  have h1 : petya_age = 3 * anya_age := by
    sorry
  have h2 : petya_age - anya_age = 8 := by
    sorry
  exact ⟨h1, h2⟩ -- Combine both conditions into a single conjunction

end ages_correct_l611_61120


namespace perpendicular_vectors_relation_l611_61106

theorem perpendicular_vectors_relation (a b : ℝ) (h : 3 * a - 7 * b = 0) : a = 7 * b / 3 :=
by
  sorry

end perpendicular_vectors_relation_l611_61106


namespace overtime_pay_rate_increase_l611_61186

theorem overtime_pay_rate_increase
  (regular_rate : ℝ)
  (total_compensation : ℝ)
  (total_hours : ℝ)
  (overtime_hours : ℝ)
  (expected_percentage_increase : ℝ)
  (h1 : regular_rate = 16)
  (h2 : total_hours = 48)
  (h3 : total_compensation = 864)
  (h4 : overtime_hours = total_hours - 40)
  (h5 : 40 * regular_rate + overtime_hours * (regular_rate + regular_rate * expected_percentage_increase / 100) = total_compensation) :
  expected_percentage_increase = 75 := 
by
  sorry

end overtime_pay_rate_increase_l611_61186


namespace magician_red_marbles_taken_l611_61110

theorem magician_red_marbles_taken:
  ∃ R : ℕ, (20 - R) + (30 - 4 * R) = 35 ∧ R = 3 :=
by
  sorry

end magician_red_marbles_taken_l611_61110


namespace feta_price_calculation_l611_61158

noncomputable def feta_price_per_pound (sandwiches_price : ℝ) (sandwiches_count : ℕ) 
  (salami_price : ℝ) (brie_factor : ℝ) (olive_price_per_pound : ℝ) 
  (olive_weight : ℝ) (bread_price : ℝ) (total_spent : ℝ)
  (feta_weight : ℝ) :=
  (total_spent - (sandwiches_count * sandwiches_price + salami_price + brie_factor * salami_price + olive_price_per_pound * olive_weight + bread_price)) / feta_weight

theorem feta_price_calculation : 
  feta_price_per_pound 7.75 2 4.00 3 10.00 0.25 2.00 40.00 0.5 = 8.00 := 
by
  sorry

end feta_price_calculation_l611_61158


namespace min_value_frac_l611_61133

theorem min_value_frac (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  ∃ (min : ℝ), min = 9 / 2 ∧ (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 2 → 4 / x + 1 / y ≥ min) :=
by
  sorry

end min_value_frac_l611_61133


namespace smallest_number_of_rectangles_l611_61150

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l611_61150


namespace find_divisor_l611_61122

theorem find_divisor (D Q R Div : ℕ) (h1 : Q = 40) (h2 : R = 64) (h3 : Div = 2944) 
  (h4 : Div = (D * Q) + R) : D = 72 :=
by
  sorry

end find_divisor_l611_61122


namespace general_term_formula_sum_first_n_terms_l611_61176

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, a n = 3^(n-2) := 
by
  sorry

theorem sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, S n = (3^(n-2)) / 2 - 1 / 18 := 
by
  sorry

end general_term_formula_sum_first_n_terms_l611_61176


namespace solve_x_for_collinear_and_same_direction_l611_61141

-- Define vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (-1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (-x, 2)

-- Define the conditions for collinearity and same direction
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k • b.1, k • b.2)

def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k • b.1, k • b.2)

theorem solve_x_for_collinear_and_same_direction
  (x : ℝ)
  (h_collinear : collinear (vector_a x) (vector_b x))
  (h_same_direction : same_direction (vector_a x) (vector_b x)) :
  x = Real.sqrt 2 :=
sorry

end solve_x_for_collinear_and_same_direction_l611_61141


namespace almonds_addition_l611_61192

theorem almonds_addition (walnuts almonds total_nuts : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_total_nuts : total_nuts = 0.5)
  (h_sum : total_nuts = walnuts + almonds) : 
  almonds = 0.25 := by
  sorry

end almonds_addition_l611_61192


namespace total_weight_of_settings_l611_61172

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l611_61172


namespace manufacturing_section_degrees_l611_61165

variable (percentage_manufacturing : ℝ) (total_degrees : ℝ)

theorem manufacturing_section_degrees
  (h1 : percentage_manufacturing = 0.40)
  (h2 : total_degrees = 360) :
  percentage_manufacturing * total_degrees = 144 := 
by 
  sorry

end manufacturing_section_degrees_l611_61165


namespace multiply_polynomials_l611_61155

variable {x y z : ℝ}

theorem multiply_polynomials :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2)
  = 27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by {
  sorry
}

end multiply_polynomials_l611_61155


namespace rug_floor_coverage_l611_61164

/-- A rectangular rug with side lengths of 2 feet and 7 feet is placed on an irregularly-shaped floor composed of a square with an area of 36 square feet and a right triangle adjacent to one of the square's sides, with leg lengths of 6 feet and 4 feet. If the surface of the rug does not extend beyond the area of the floor, then the fraction of the area of the floor that is not covered by the rug is 17/24. -/
theorem rug_floor_coverage : (48 - 14) / 48 = 17 / 24 :=
by
  -- proof goes here
  sorry

end rug_floor_coverage_l611_61164


namespace max_value_of_expression_l611_61142

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h_sum : x + y + z = 3) 
  (h_order : x ≥ y ∧ y ≥ z) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 :=
sorry

end max_value_of_expression_l611_61142


namespace charlie_cost_per_gb_l611_61190

noncomputable def total_data_usage (w1 w2 w3 w4 : ℕ) : ℕ := w1 + w2 + w3 + w4

noncomputable def data_over_limit (total_data usage_limit: ℕ) : ℕ :=
  if total_data > usage_limit then total_data - usage_limit else 0

noncomputable def cost_per_gb (extra_cost data_over_limit: ℕ) : ℕ :=
  if data_over_limit > 0 then extra_cost / data_over_limit else 0

theorem charlie_cost_per_gb :
  let D := 8
  let w1 := 2
  let w2 := 3
  let w3 := 5
  let w4 := 10
  let C := 120
  let total_data := total_data_usage w1 w2 w3 w4
  let data_over := data_over_limit total_data D
  C / data_over = 10 := by
  -- Sorry to skip the proof
  sorry

end charlie_cost_per_gb_l611_61190


namespace ellipse_tangent_line_l611_61185

theorem ellipse_tangent_line (m : ℝ) : 
  (∀ (x y : ℝ), (x ^ 2 / 4) + (y ^ 2 / m) = 1 → (y = mx + 2)) → m = 1 :=
by sorry

end ellipse_tangent_line_l611_61185


namespace line_canonical_form_l611_61148

theorem line_canonical_form :
  (∀ x y z : ℝ, 4 * x + y - 3 * z + 2 = 0 → 2 * x - y + z - 8 = 0 ↔
    ∃ t : ℝ, x = 1 + -2 * t ∧ y = -6 + -10 * t ∧ z = -6 * t) :=
by
  sorry

end line_canonical_form_l611_61148


namespace find_constant_l611_61102

-- Define the relationship between Fahrenheit and Celsius
def temp_rel (c f k : ℝ) : Prop :=
  f = (9 / 5) * c + k

-- Temperature increases
def temp_increase (c1 c2 f1 f2 : ℝ) : Prop :=
  (f2 - f1 = 30) ∧ (c2 - c1 = 16.666666666666668)

-- Freezing point condition
def freezing_point (f : ℝ) : Prop :=
  f = 32

-- Main theorem to prove
theorem find_constant (k : ℝ) :
  ∃ (c1 c2 f1 f2: ℝ), temp_rel c1 f1 k ∧ temp_rel c2 f2 k ∧ 
  temp_increase c1 c2 f1 f2 ∧ freezing_point f1 → k = 32 :=
by sorry

end find_constant_l611_61102


namespace son_is_four_times_younger_l611_61169

-- Given Conditions
def son_age : ℕ := 9
def dad_age : ℕ := 36
def age_difference : ℕ := dad_age - son_age -- Ensure the difference in ages

-- The proof problem
theorem son_is_four_times_younger : dad_age / son_age = 4 :=
by
  -- Ensure the conditions are correct and consistent.
  have h1 : dad_age = 36 := rfl
  have h2 : son_age = 9 := rfl
  have h3 : dad_age - son_age = 27 := rfl
  sorry

end son_is_four_times_younger_l611_61169


namespace find_m_if_f_monotonic_l611_61137

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  4 * x^3 + m * x^2 + (m - 3) * x + n

def is_monotonically_increasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2

theorem find_m_if_f_monotonic (m n : ℝ)
  (h : is_monotonically_increasing_on_ℝ (f m n)) :
  m = 6 :=
sorry

end find_m_if_f_monotonic_l611_61137


namespace remainder_of_173_mod_13_l611_61131

theorem remainder_of_173_mod_13 : ∀ (m : ℤ), 173 = 8 * m + 5 → 173 < 180 → 173 % 13 = 4 :=
by
  intro m hm h
  sorry

end remainder_of_173_mod_13_l611_61131


namespace find_shaun_age_l611_61178

def current_ages (K G S : ℕ) :=
  K + 4 = 2 * (G + 4) ∧
  S + 8 = 2 * (K + 8) ∧
  S + 12 = 3 * (G + 12)

theorem find_shaun_age (K G S : ℕ) (h : current_ages K G S) : S = 48 :=
  by
    sorry

end find_shaun_age_l611_61178


namespace combined_loading_time_l611_61193

theorem combined_loading_time (rA rB rC : ℝ) (hA : rA = 1 / 6) (hB : rB = 1 / 8) (hC : rC = 1 / 10) :
  1 / (rA + rB + rC) = 120 / 47 := by
  sorry

end combined_loading_time_l611_61193


namespace annual_rent_per_square_foot_l611_61136

theorem annual_rent_per_square_foot 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (area : ℕ)
  (annual_rent : ℕ) : 
  monthly_rent = 3600 → 
  length = 18 → 
  width = 20 → 
  area = length * width → 
  annual_rent = monthly_rent * 12 → 
  annual_rent / area = 120 :=
by
  sorry

end annual_rent_per_square_foot_l611_61136


namespace geometric_progression_ratio_l611_61166

theorem geometric_progression_ratio (r : ℕ) (h : 4 + 4 * r + 4 * r^2 + 4 * r^3 = 60) : r = 2 :=
by
  sorry

end geometric_progression_ratio_l611_61166


namespace rectangle_percentage_increase_l611_61127

theorem rectangle_percentage_increase (L W : ℝ) (P : ℝ) (h : (1 + P / 100) ^ 2 = 1.44) : P = 20 :=
by {
  -- skipped proof
  sorry
}

end rectangle_percentage_increase_l611_61127


namespace work_completed_in_8_days_l611_61175

theorem work_completed_in_8_days 
  (A_complete : ℕ → Prop)
  (B_complete : ℕ → Prop)
  (C_complete : ℕ → Prop)
  (A_can_complete_in_10_days : A_complete 10)
  (B_can_complete_in_20_days : B_complete 20)
  (C_can_complete_in_30_days : C_complete 30)
  (A_leaves_5_days_before_completion : ∀ x : ℕ, x ≥ 5 → A_complete (x - 5))
  (C_leaves_3_days_before_completion : ∀ x : ℕ, x ≥ 3 → C_complete (x - 3)) :
  ∃ x : ℕ, x = 8 := sorry

end work_completed_in_8_days_l611_61175


namespace find_number_eq_seven_point_five_l611_61167

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l611_61167


namespace flowerbed_seeds_l611_61171

theorem flowerbed_seeds (n_fbeds n_seeds_per_fbed total_seeds : ℕ)
    (h1 : n_fbeds = 8)
    (h2 : n_seeds_per_fbed = 4) :
    total_seeds = n_fbeds * n_seeds_per_fbed := by
  sorry

end flowerbed_seeds_l611_61171


namespace rich_total_distance_l611_61144

-- Define the given conditions 
def distance_house_to_sidewalk := 20
def distance_down_road := 200
def total_distance_so_far := distance_house_to_sidewalk + distance_down_road
def distance_left_turn := 2 * total_distance_so_far
def distance_to_intersection := total_distance_so_far + distance_left_turn
def distance_half := distance_to_intersection / 2
def total_distance_one_way := distance_to_intersection + distance_half

-- Define the theorem to be proven 
theorem rich_total_distance : total_distance_one_way * 2 = 1980 :=
by 
  -- This line is to complete the 'prove' demand of the theorem
  sorry

end rich_total_distance_l611_61144


namespace fraction_of_students_participated_l611_61105

theorem fraction_of_students_participated (total_students : ℕ) (did_not_participate : ℕ)
  (h_total : total_students = 39) (h_did_not_participate : did_not_participate = 26) :
  (total_students - did_not_participate) / total_students = 1 / 3 :=
by
  sorry

end fraction_of_students_participated_l611_61105


namespace chorus_group_membership_l611_61188

theorem chorus_group_membership (n : ℕ) : 
  100 < n ∧ n < 200 →
  n % 3 = 1 ∧ 
  n % 4 = 2 ∧ 
  n % 6 = 4 ∧ 
  n % 8 = 6 →
  n = 118 ∨ n = 142 ∨ n = 166 ∨ n = 190 :=
by
  sorry

end chorus_group_membership_l611_61188


namespace smallest_number_l611_61177

-- Definitions based on the conditions given in the problem
def satisfies_conditions (b : ℕ) : Prop :=
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1

-- Lean proof statement
theorem smallest_number (b : ℕ) : satisfies_conditions b → b = 87 :=
sorry

end smallest_number_l611_61177


namespace solve_for_x_l611_61135

-- Definitions based on provided conditions
variables (x : ℝ) -- defining x as a real number
def condition : Prop := 0.25 * x = 0.15 * 1600 - 15

-- The theorem stating that x equals 900 given the condition
theorem solve_for_x (h : condition x) : x = 900 :=
by
  sorry

end solve_for_x_l611_61135


namespace quadratic_solve_l611_61134

theorem quadratic_solve (x : ℝ) : (x + 4)^2 = 5 * (x + 4) → x = -4 ∨ x = 1 :=
by sorry

end quadratic_solve_l611_61134


namespace a2020_lt_inv_2020_l611_61174

theorem a2020_lt_inv_2020 (a : ℕ → ℝ) (ha0 : a 0 > 0) 
    (h_rec : ∀ n, a (n + 1) = a n / Real.sqrt (1 + 2020 * a n ^ 2)) :
    a 2020 < 1 / 2020 :=
sorry

end a2020_lt_inv_2020_l611_61174


namespace triangle_side_lengths_exist_l611_61126

theorem triangle_side_lengths_exist :
  ∃ (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ abc = 2 * (a - 1) * (b - 1) * (c - 1) ∧
  ((a, b, c) = (8, 7, 3) ∨ (a, b, c) = (6, 5, 4)) :=
by sorry

end triangle_side_lengths_exist_l611_61126


namespace find_k_value_l611_61184

theorem find_k_value (x k : ℝ) (h : x = -3) (h_eq : k * (x - 2) - 4 = k - 2 * x) : k = -5/3 := by
  sorry

end find_k_value_l611_61184


namespace pinky_pies_count_l611_61198

theorem pinky_pies_count (helen_pies : ℕ) (total_pies : ℕ) (h1 : helen_pies = 56) (h2 : total_pies = 203) : 
  total_pies - helen_pies = 147 := by
  sorry

end pinky_pies_count_l611_61198


namespace find_integer_divisible_by_18_and_square_root_in_range_l611_61125

theorem find_integer_divisible_by_18_and_square_root_in_range :
  ∃ x : ℕ, 28 < Real.sqrt x ∧ Real.sqrt x < 28.2 ∧ 18 ∣ x ∧ x = 792 :=
by
  sorry

end find_integer_divisible_by_18_and_square_root_in_range_l611_61125


namespace problem1_problem2_problem3_problem4_l611_61199

theorem problem1 : -20 - (-14) + (-18) - 13 = -37 := by
  sorry

theorem problem2 : (-3/4 + 1/6 - 5/8) / (-1/24) = 29 := by
  sorry

theorem problem3 : -3^2 + (-3)^2 + 3 * 2 + |(-4)| = 10 := by
  sorry

theorem problem4 : 16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3 := by
  sorry

end problem1_problem2_problem3_problem4_l611_61199


namespace sequence_a100_l611_61149

theorem sequence_a100 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ m n : ℕ, 0 < m → 0 < n → a (n + m) = a n + a m + n * m) ∧ (a 100 = 5050) :=
by
  sorry

end sequence_a100_l611_61149


namespace cucumber_weight_evaporation_l611_61154

theorem cucumber_weight_evaporation :
  let w_99 := 50
  let p_99 := 0.99
  let evap_99 := 0.01
  let w_98 := 30
  let p_98 := 0.98
  let evap_98 := 0.02
  let w_97 := 20
  let p_97 := 0.97
  let evap_97 := 0.03

  let initial_water_99 := p_99 * w_99
  let dry_matter_99 := w_99 - initial_water_99
  let evaporated_water_99 := evap_99 * initial_water_99
  let new_weight_99 := (initial_water_99 - evaporated_water_99) + dry_matter_99

  let initial_water_98 := p_98 * w_98
  let dry_matter_98 := w_98 - initial_water_98
  let evaporated_water_98 := evap_98 * initial_water_98
  let new_weight_98 := (initial_water_98 - evaporated_water_98) + dry_matter_98

  let initial_water_97 := p_97 * w_97
  let dry_matter_97 := w_97 - initial_water_97
  let evaporated_water_97 := evap_97 * initial_water_97
  let new_weight_97 := (initial_water_97 - evaporated_water_97) + dry_matter_97

  let total_new_weight := new_weight_99 + new_weight_98 + new_weight_97
  total_new_weight = 98.335 :=
 by
  sorry

end cucumber_weight_evaporation_l611_61154


namespace factorization_sum_l611_61124

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x^2 + 9 * x + 20 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 21 :=
by
  sorry

end factorization_sum_l611_61124


namespace small_bottles_sold_percentage_l611_61160

theorem small_bottles_sold_percentage
  (small_bottles : ℕ) (big_bottles : ℕ) (percent_sold_big_bottles : ℝ)
  (remaining_bottles : ℕ) (percent_sold_small_bottles : ℝ) :
  small_bottles = 6000 ∧
  big_bottles = 14000 ∧
  percent_sold_big_bottles = 0.23 ∧
  remaining_bottles = 15580 ∧ 
  percent_sold_small_bottles / 100 * 6000 + 0.23 * 14000 + remaining_bottles = small_bottles + big_bottles →
  percent_sold_small_bottles = 37 := 
by
  intros
  exact sorry

end small_bottles_sold_percentage_l611_61160


namespace yield_is_eight_percent_l611_61119

noncomputable def par_value : ℝ := 100
noncomputable def annual_dividend : ℝ := 0.12 * par_value
noncomputable def market_value : ℝ := 150
noncomputable def yield_percentage : ℝ := (annual_dividend / market_value) * 100

theorem yield_is_eight_percent : yield_percentage = 8 := 
by 
  sorry

end yield_is_eight_percent_l611_61119


namespace two_digit_subtraction_pattern_l611_61111

theorem two_digit_subtraction_pattern (a b : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) := 
by
  sorry

end two_digit_subtraction_pattern_l611_61111


namespace trig_identity_problem_l611_61128

theorem trig_identity_problem
  (x : ℝ) (a b c : ℕ)
  (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.sin x - Real.cos x = Real.pi / 4)
  (h3 : Real.tan x + 1 / Real.tan x = (a : ℝ) / (b - Real.pi^c)) :
  a + b + c = 50 :=
sorry

end trig_identity_problem_l611_61128


namespace incorrect_statement_proof_l611_61180

-- Define the conditions as assumptions
def inductive_reasoning_correct : Prop := ∀ (P : Prop), ¬(P → P)
def analogical_reasoning_correct : Prop := ∀ (P Q : Prop), ¬(P → Q)
def reasoning_by_plausibility_correct : Prop := ∀ (P : Prop), ¬(P → P)

-- Define the incorrect statement to be proven
def inductive_reasoning_incorrect_statement : Prop := 
  ¬ (∀ (P Q : Prop), ¬(P ↔ Q))

-- The theorem to be proven
theorem incorrect_statement_proof 
  (h1 : inductive_reasoning_correct)
  (h2 : analogical_reasoning_correct)
  (h3 : reasoning_by_plausibility_correct) : inductive_reasoning_incorrect_statement :=
sorry

end incorrect_statement_proof_l611_61180


namespace pig_count_correct_l611_61113

def initial_pigs : ℝ := 64.0
def additional_pigs : ℝ := 86.0
def total_pigs : ℝ := 150.0

theorem pig_count_correct : initial_pigs + additional_pigs = total_pigs := by
  show 64.0 + 86.0 = 150.0
  sorry

end pig_count_correct_l611_61113


namespace imaginary_part_of_i_mul_root_l611_61195

theorem imaginary_part_of_i_mul_root
  (z : ℂ) (hz : z^2 - 4 * z + 5 = 0) : (i * z).im = 2 := 
sorry

end imaginary_part_of_i_mul_root_l611_61195


namespace gcd_fact_plus_two_l611_61159

theorem gcd_fact_plus_two (n m : ℕ) (h1 : n = 6) (h2 : m = 8) :
  Nat.gcd (n.factorial + 2) (m.factorial + 2) = 2 :=
  sorry

end gcd_fact_plus_two_l611_61159
