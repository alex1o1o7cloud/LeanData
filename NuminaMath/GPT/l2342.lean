import Mathlib

namespace NUMINAMATH_GPT_cost_of_patent_is_correct_l2342_234271

-- Defining the conditions
def c_parts : ℕ := 3600
def p : ℕ := 180
def n : ℕ := 45

-- Calculation of total revenue
def total_revenue : ℕ := n * p

-- Calculation of cost of patent
def cost_of_patent (total_revenue c_parts : ℕ) : ℕ := total_revenue - c_parts

-- The theorem to be proved
theorem cost_of_patent_is_correct (R : ℕ) (H : R = total_revenue) : cost_of_patent R c_parts = 4500 :=
by
  -- this is where your proof will go
  sorry

end NUMINAMATH_GPT_cost_of_patent_is_correct_l2342_234271


namespace NUMINAMATH_GPT_tommy_balloons_l2342_234247

/-- Tommy had some balloons. He received 34 more balloons from his mom,
gave away 15 balloons, and exchanged the remaining balloons for teddy bears
at a rate of 3 balloons per teddy bear. After these transactions, he had 30 teddy bears.
Prove that Tommy started with 71 balloons -/
theorem tommy_balloons : 
  ∃ B : ℕ, (B + 34 - 15) = 3 * 30 ∧ B = 71 := 
by
  have h : (71 + 34 - 15) = 3 * 30 := by norm_num
  exact ⟨71, h, rfl⟩

end NUMINAMATH_GPT_tommy_balloons_l2342_234247


namespace NUMINAMATH_GPT_determine_defective_coin_l2342_234289

-- Define the properties of the coins
structure Coin :=
(denomination : ℕ)
(weight : ℕ)

-- Given coins
def c1 : Coin := ⟨1, 1⟩
def c2 : Coin := ⟨2, 2⟩
def c3 : Coin := ⟨3, 3⟩
def c5 : Coin := ⟨5, 5⟩

-- Assume one coin is defective
variable (defective : Coin)
variable (differing_weight : ℕ)
#check differing_weight

theorem determine_defective_coin :
  (∃ (defective : Coin), ∀ (c : Coin), 
    c ≠ defective → c.weight = c.denomination) → 
  ((c2.weight + c3.weight = c5.weight → defective = c1) ∧
   (c1.weight + c2.weight = c3.weight → defective = c5) ∧
   (c2.weight ≠ 2 → defective = c2) ∧
   (c3.weight ≠ 3 → defective = c3)) :=
by
  sorry

end NUMINAMATH_GPT_determine_defective_coin_l2342_234289


namespace NUMINAMATH_GPT_orangeade_price_second_day_l2342_234250

theorem orangeade_price_second_day :
  ∀ (X O : ℝ), (2 * X * 0.60 = 3 * X * E) → (E = 2 * 0.60 / 3) →
  E = 0.40 := by
  intros X O h₁ h₂
  sorry

end NUMINAMATH_GPT_orangeade_price_second_day_l2342_234250


namespace NUMINAMATH_GPT_solve_part_a_solve_part_b_solve_part_c_l2342_234285

-- Part (a)
theorem solve_part_a (x : ℝ) : 
  (2 * x^2 + 3 * x - 1)^2 - 5 * (2 * x^2 + 3 * x + 3) + 24 = 0 ↔ 
  x = 1 ∨ x = -2 ∨ x = 0.5 ∨ x = -2.5 := sorry

-- Part (b)
theorem solve_part_b (x : ℝ) : 
  (x - 1) * (x + 3) * (x + 4) * (x + 8) = -96 ↔ 
  x = 0 ∨ x = -7 ∨ x = (-7 + Real.sqrt 33) / 2 ∨ x = (-7 - Real.sqrt 33) / 2 := sorry

-- Part (c)
theorem solve_part_c (x : ℝ) (hx : x ≠ 0) : 
  (x - 1) * (x - 2) * (x - 4) * (x - 8) = 4 * x^2 ↔ 
  x = 4 + 2 * Real.sqrt 2 ∨ x = 4 - 2 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_solve_part_a_solve_part_b_solve_part_c_l2342_234285


namespace NUMINAMATH_GPT_triangle_pentagon_side_ratio_l2342_234290

theorem triangle_pentagon_side_ratio (triangle_perimeter : ℕ) (pentagon_perimeter : ℕ) 
  (h1 : triangle_perimeter = 60) (h2 : pentagon_perimeter = 60) :
  (triangle_perimeter / 3 : ℚ) / (pentagon_perimeter / 5 : ℚ) = 5 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_pentagon_side_ratio_l2342_234290


namespace NUMINAMATH_GPT_range_of_a_l2342_234255

open Set Real

theorem range_of_a (a : ℝ) (α : ℝ → Prop) (β : ℝ → Prop) (hα : ∀ x, α x ↔ x ≥ a) (hβ : ∀ x, β x ↔ |x - 1| < 1)
  (h : ∀ x, (β x → α x) ∧ (∃ x, α x ∧ ¬β x)) : a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2342_234255


namespace NUMINAMATH_GPT_total_cost_correct_l2342_234242

-- Defining the conditions
def charges_per_week : ℕ := 3
def weeks_per_year : ℕ := 52
def cost_per_charge : ℝ := 0.78

-- Defining the total cost proof statement
theorem total_cost_correct : (charges_per_week * weeks_per_year : ℝ) * cost_per_charge = 121.68 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l2342_234242


namespace NUMINAMATH_GPT_plate_729_driving_days_l2342_234256

def plate (n : ℕ) : Prop := n >= 0 ∧ n <= 999

def monday (n : ℕ) : Prop := n % 2 = 1

def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3

def tuesday (n : ℕ) : Prop := sum_digits n >= 11

def wednesday (n : ℕ) : Prop := n % 3 = 0

def thursday (n : ℕ) : Prop := sum_digits n <= 14

def count_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100, (n / 10) % 10, n % 10)

def friday (n : ℕ) : Prop :=
  let (d1, d2, d3) := count_digits n
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

def saturday (n : ℕ) : Prop := n < 500

def sunday (n : ℕ) : Prop := 
  let (d1, d2, d3) := count_digits n
  d1 <= 5 ∧ d2 <= 5 ∧ d3 <= 5

def can_drive (n : ℕ) (day : String) : Prop :=
  plate n ∧ 
  (day = "Monday" → monday n) ∧ 
  (day = "Tuesday" → tuesday n) ∧ 
  (day = "Wednesday" → wednesday n) ∧ 
  (day = "Thursday" → thursday n) ∧ 
  (day = "Friday" → friday n) ∧ 
  (day = "Saturday" → saturday n) ∧ 
  (day = "Sunday" → sunday n)

theorem plate_729_driving_days :
  can_drive 729 "Monday" ∧
  can_drive 729 "Tuesday" ∧
  can_drive 729 "Wednesday" ∧
  ¬ can_drive 729 "Thursday" ∧
  ¬ can_drive 729 "Friday" ∧
  ¬ can_drive 729 "Saturday" ∧
  ¬ can_drive 729 "Sunday" :=
by
  sorry

end NUMINAMATH_GPT_plate_729_driving_days_l2342_234256


namespace NUMINAMATH_GPT_moles_of_water_formed_l2342_234223

-- Defining the relevant constants
def NH4Cl_moles : ℕ := sorry  -- Some moles of Ammonium chloride (NH4Cl)
def NaOH_moles : ℕ := 3       -- 3 moles of Sodium hydroxide (NaOH)
def H2O_moles : ℕ := 3        -- The total moles of Water (H2O) formed

-- Statement of the problem
theorem moles_of_water_formed :
  NH4Cl_moles ≥ NaOH_moles → H2O_moles = 3 :=
sorry

end NUMINAMATH_GPT_moles_of_water_formed_l2342_234223


namespace NUMINAMATH_GPT_gcd_3pow600_minus_1_3pow612_minus_1_l2342_234212

theorem gcd_3pow600_minus_1_3pow612_minus_1 :
  Nat.gcd (3^600 - 1) (3^612 - 1) = 531440 :=
by
  sorry

end NUMINAMATH_GPT_gcd_3pow600_minus_1_3pow612_minus_1_l2342_234212


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l2342_234239

variable (x : ℝ)

theorem speed_of_boat_in_still_water (h : 10 = (x + 5) * 0.4) : x = 20 :=
sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l2342_234239


namespace NUMINAMATH_GPT_pool_people_count_l2342_234288

theorem pool_people_count (P : ℕ) (total_money : ℝ) (cost_per_person : ℝ) (leftover_money : ℝ) 
  (h1 : total_money = 30) 
  (h2 : cost_per_person = 2.50) 
  (h3 : leftover_money = 5) 
  (h4 : total_money - leftover_money = cost_per_person * P) : 
  P = 10 :=
sorry

end NUMINAMATH_GPT_pool_people_count_l2342_234288


namespace NUMINAMATH_GPT_sin_750_eq_one_half_l2342_234210

theorem sin_750_eq_one_half :
  ∀ (θ: ℝ), (∀ n: ℤ, Real.sin (θ + n * 360) = Real.sin θ) → Real.sin 30 = 1 / 2 → Real.sin 750 = 1 / 2 :=
by 
  intros θ periodic_sine sin_30
  -- insert proof here
  sorry

end NUMINAMATH_GPT_sin_750_eq_one_half_l2342_234210


namespace NUMINAMATH_GPT_next_term_geometric_sequence_l2342_234259

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end NUMINAMATH_GPT_next_term_geometric_sequence_l2342_234259


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l2342_234270

theorem inscribed_sphere_radius (b d : ℝ) : 
  (b * Real.sqrt d - b = 15 * (Real.sqrt 5 - 1) / 4) → 
  b + d = 11.75 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l2342_234270


namespace NUMINAMATH_GPT_base_subtraction_proof_l2342_234222

def convert_base8_to_base10 (n : Nat) : Nat :=
  5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1

def convert_base9_to_base10 (n : Nat) : Nat :=
  4 * 9^3 + 3 * 9^2 + 2 * 9^1 + 1

theorem base_subtraction_proof :
  convert_base8_to_base10 54321 - convert_base9_to_base10 4321 = 19559 :=
by
  sorry

end NUMINAMATH_GPT_base_subtraction_proof_l2342_234222


namespace NUMINAMATH_GPT_comprehensive_survey_is_C_l2342_234229

def option (label : String) (description : String) := (label, description)

def A := option "A" "Investigating the current mental health status of middle school students nationwide"
def B := option "B" "Investigating the compliance of food in our city"
def C := option "C" "Investigating the physical and mental conditions of classmates in the class"
def D := option "D" "Investigating the viewership ratings of Nanjing TV's 'Today's Life'"

theorem comprehensive_survey_is_C (suitable: (String × String → Prop)) :
  suitable C :=
sorry

end NUMINAMATH_GPT_comprehensive_survey_is_C_l2342_234229


namespace NUMINAMATH_GPT_annual_decrease_rate_l2342_234235

def initial_population : ℝ := 8000
def population_after_two_years : ℝ := 3920

theorem annual_decrease_rate :
  ∃ r : ℝ, (0 < r ∧ r < 1) ∧ (initial_population * (1 - r)^2 = population_after_two_years) ∧ r = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_annual_decrease_rate_l2342_234235


namespace NUMINAMATH_GPT_smallest_spherical_triangle_angle_l2342_234274

-- Define the conditions
def is_ratio (a b c : ℕ) : Prop := a = 4 ∧ b = 5 ∧ c = 6
def sum_of_angles (α β γ : ℕ) : Prop := α + β + γ = 270

-- Define the problem statement
theorem smallest_spherical_triangle_angle 
  (a b c α β γ : ℕ)
  (h1 : is_ratio a b c)
  (h2 : sum_of_angles (a * α) (b * β) (c * γ)) :
  a * α = 72 := 
sorry

end NUMINAMATH_GPT_smallest_spherical_triangle_angle_l2342_234274


namespace NUMINAMATH_GPT_simplify_radical_subtraction_l2342_234284

theorem simplify_radical_subtraction : 
  (Real.sqrt 18 - Real.sqrt 8) = Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_radical_subtraction_l2342_234284


namespace NUMINAMATH_GPT_sqrt_range_l2342_234246

theorem sqrt_range (x : ℝ) : 3 - 2 * x ≥ 0 ↔ x ≤ 3 / 2 := 
    sorry

end NUMINAMATH_GPT_sqrt_range_l2342_234246


namespace NUMINAMATH_GPT_option_D_correct_l2342_234267

variable (x : ℝ)

theorem option_D_correct : (2 * x^7) / x = 2 * x^6 := sorry

end NUMINAMATH_GPT_option_D_correct_l2342_234267


namespace NUMINAMATH_GPT_crayons_allocation_correct_l2342_234268

noncomputable def crayons_allocation : Prop :=
  ∃ (F B J S : ℕ), 
    F + B + J + S = 96 ∧ 
    F = 2 * B ∧ 
    J = 3 * S ∧ 
    B = 12 ∧ 
    F = 24 ∧ 
    J = 45 ∧ 
    S = 15

theorem crayons_allocation_correct : crayons_allocation :=
  sorry

end NUMINAMATH_GPT_crayons_allocation_correct_l2342_234268


namespace NUMINAMATH_GPT_circle_intersection_range_l2342_234287

noncomputable def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
noncomputable def circle2_eq (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

theorem circle_intersection_range (r : ℝ) (h : r > 0) :
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y r) ↔ 2 < r ∧ r < 12 :=
sorry

end NUMINAMATH_GPT_circle_intersection_range_l2342_234287


namespace NUMINAMATH_GPT_eight_b_equals_neg_eight_l2342_234237

theorem eight_b_equals_neg_eight (a b : ℤ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := 
by
  sorry

end NUMINAMATH_GPT_eight_b_equals_neg_eight_l2342_234237


namespace NUMINAMATH_GPT_cuboid_surface_area_correct_l2342_234266

-- Define the dimensions of the cuboid
def l : ℕ := 4
def w : ℕ := 5
def h : ℕ := 6

-- Define the function to calculate the surface area of the cuboid
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

-- The theorem stating that the surface area of the cuboid is 148 cm²
theorem cuboid_surface_area_correct : surface_area l w h = 148 := by
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_correct_l2342_234266


namespace NUMINAMATH_GPT_correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l2342_234265

-- Define the constants k and b
variables (k b : ℝ)

-- Define the function y = k * t + b
def linear_func (t : ℝ) : ℝ := k * t + b

-- Define the data points as conditions
axiom data_point1 : linear_func k b 1 = 7
axiom data_point2 : linear_func k b 2 = 12
axiom data_point3 : linear_func k b 3 = 17
axiom data_point4 : linear_func k b 4 = 22
axiom data_point5 : linear_func k b 5 = 27

-- Define the water consumption rate and total minutes in a day
def daily_water_consumption : ℝ := 1500
def minutes_in_one_day : ℝ := 1440
def days_in_month : ℝ := 30

-- The expression y = 5t + 2
theorem correct_functional_relationship : (k = 5) ∧ (b = 2) :=
by
  sorry

-- Estimated water amount at the 20th minute
theorem water_amount_20th_minute (t : ℝ) (ht : t = 20) : linear_func 5 2 t = 102 :=
by
  sorry

-- The water leaked in a month (30 days) can supply the number of days
theorem water_amount_supply_days : (linear_func 5 2 (minutes_in_one_day * days_in_month)) / daily_water_consumption = 144 :=
by
  sorry

end NUMINAMATH_GPT_correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l2342_234265


namespace NUMINAMATH_GPT_single_colony_reaches_limit_in_24_days_l2342_234204

/-- A bacteria colony doubles in size every day. -/
def double (n : ℕ) : ℕ := 2 ^ n

/-- Two bacteria colonies growing simultaneously will take 24 days to reach the habitat's limit. -/
axiom two_colonies_24_days : ∀ k : ℕ, double k + double k = double 24

/-- Prove that it takes 24 days for a single bacteria colony to reach the habitat's limit. -/
theorem single_colony_reaches_limit_in_24_days : ∃ x : ℕ, double x = double 24 :=
sorry

end NUMINAMATH_GPT_single_colony_reaches_limit_in_24_days_l2342_234204


namespace NUMINAMATH_GPT_gain_percent_l2342_234216

theorem gain_percent (C S : ℝ) (h : 50 * C = 15 * S) :
  (S > C) →
  ((S - C) / C * 100) = 233.33 := 
sorry

end NUMINAMATH_GPT_gain_percent_l2342_234216


namespace NUMINAMATH_GPT_greatest_median_l2342_234207

theorem greatest_median (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t) (h5 : (k + m + r + s + t) = 80) (h6 : t = 42) : r = 17 :=
by
  sorry

end NUMINAMATH_GPT_greatest_median_l2342_234207


namespace NUMINAMATH_GPT_balls_into_boxes_all_ways_balls_into_boxes_one_empty_l2342_234225

/-- There are 4 different balls and 4 different boxes. -/
def balls : ℕ := 4
def boxes : ℕ := 4

/-- The number of ways to put 4 different balls into 4 different boxes is 256. -/
theorem balls_into_boxes_all_ways : (balls ^ boxes) = 256 := by
  sorry

/-- The number of ways to put 4 different balls into 4 different boxes such that exactly one box remains empty is 144. -/
theorem balls_into_boxes_one_empty : (boxes.choose 1 * (balls ^ (boxes - 1))) = 144 := by
  sorry

end NUMINAMATH_GPT_balls_into_boxes_all_ways_balls_into_boxes_one_empty_l2342_234225


namespace NUMINAMATH_GPT_cost_of_paving_l2342_234278

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 1400
def expected_cost : ℝ := 28875

theorem cost_of_paving (l w r : ℝ) (h_l : l = length) (h_w : w = width) (h_r : r = rate_per_sqm) :
  (l * w * r) = expected_cost := by
  sorry

end NUMINAMATH_GPT_cost_of_paving_l2342_234278


namespace NUMINAMATH_GPT_multiple_of_24_l2342_234292

theorem multiple_of_24 (n : ℕ) (h : n > 0) : 
  ∃ k₁ k₂ : ℕ, (6 * n - 1)^2 - 1 = 24 * k₁ ∧ (6 * n + 1)^2 - 1 = 24 * k₂ :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_24_l2342_234292


namespace NUMINAMATH_GPT_number_of_green_pens_l2342_234230

theorem number_of_green_pens
  (black_pens : ℕ := 6)
  (red_pens : ℕ := 7)
  (green_pens : ℕ)
  (probability_black : (black_pens : ℚ) / (black_pens + red_pens + green_pens : ℚ) = 1 / 3) :
  green_pens = 5 := 
sorry

end NUMINAMATH_GPT_number_of_green_pens_l2342_234230


namespace NUMINAMATH_GPT_polynomial_divisible_by_24_l2342_234282

-- Defining the function
def f (n : ℕ) : ℕ :=
n^4 + 2*n^3 + 11*n^2 + 10*n

-- Statement of the theorem
theorem polynomial_divisible_by_24 (n : ℕ) (h : n > 0) : f n % 24 = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_24_l2342_234282


namespace NUMINAMATH_GPT_eyes_per_ant_proof_l2342_234209

noncomputable def eyes_per_ant (s a e_s E : ℕ) : ℕ :=
  let e_spiders := s * e_s
  let e_ants := E - e_spiders
  e_ants / a

theorem eyes_per_ant_proof : eyes_per_ant 3 50 8 124 = 2 :=
by
  sorry

end NUMINAMATH_GPT_eyes_per_ant_proof_l2342_234209


namespace NUMINAMATH_GPT_molecular_weight_H2O_7_moles_l2342_234273

noncomputable def atomic_weight_H : ℝ := 1.008
noncomputable def atomic_weight_O : ℝ := 16.00
noncomputable def num_atoms_H_in_H2O : ℝ := 2
noncomputable def num_atoms_O_in_H2O : ℝ := 1
noncomputable def moles_H2O : ℝ := 7

theorem molecular_weight_H2O_7_moles :
  (num_atoms_H_in_H2O * atomic_weight_H + num_atoms_O_in_H2O * atomic_weight_O) * moles_H2O = 126.112 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_H2O_7_moles_l2342_234273


namespace NUMINAMATH_GPT_sum_of_angles_FC_correct_l2342_234254

noncomputable def circleGeometry (A B C D E F : Point)
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E)
  (arcAB : ℝ) (arcDE : ℝ) : Prop :=
  let arcFull := 360;
  let angleF := 6;  -- Derived from the intersecting chords theorem
  let angleC := 36; -- Derived from the inscribed angle theorem
  arcAB = 60 ∧ arcDE = 72 ∧
  0 ≤ angleF ∧ 0 ≤ angleC ∧
  angleF + angleC = 42

theorem sum_of_angles_FC_correct (A B C D E F : Point) 
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) :
  circleGeometry A B C D E F onCircle 60 72 :=
by
  sorry  -- Proof to be filled

end NUMINAMATH_GPT_sum_of_angles_FC_correct_l2342_234254


namespace NUMINAMATH_GPT_nba_conference_division_impossible_l2342_234217

theorem nba_conference_division_impossible :
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  ¬∃ (A B : ℕ), A + B = teams ∧ A * B = inter_conference_games := 
by
  let teams := 30
  let games_per_team := 82
  let total_games := teams * games_per_team
  let unique_games := total_games / 2
  let inter_conference_games := unique_games / 2
  sorry

end NUMINAMATH_GPT_nba_conference_division_impossible_l2342_234217


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_q_l2342_234281

-- Define the conditions p and q
def p (x : ℝ) : Prop := 2 < x ∧ x < 4
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

-- Prove the relationship between p and q
theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_q_l2342_234281


namespace NUMINAMATH_GPT_closest_perfect_square_to_350_l2342_234226

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end NUMINAMATH_GPT_closest_perfect_square_to_350_l2342_234226


namespace NUMINAMATH_GPT_integer_fraction_condition_l2342_234279

theorem integer_fraction_condition (p : ℕ) (h_pos : 0 < p) :
  (∃ k : ℤ, k > 0 ∧ (5 * p + 15) = k * (3 * p - 9)) ↔ (4 ≤ p ∧ p ≤ 19) :=
by
  sorry

end NUMINAMATH_GPT_integer_fraction_condition_l2342_234279


namespace NUMINAMATH_GPT_simplify_eval_expression_l2342_234258

variables (a b : ℝ)

theorem simplify_eval_expression :
  a = Real.sqrt 3 →
  b = Real.sqrt 3 - 1 →
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_eval_expression_l2342_234258


namespace NUMINAMATH_GPT_f_alpha_l2342_234297

variables (α : Real) (x : Real)

noncomputable def f (x : Real) : Real := 
  (Real.cos (Real.pi + x) * Real.sin (2 * Real.pi - x)) / Real.cos (Real.pi - x)

lemma sin_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) : 
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

lemma tan_alpha {α : Real} (hsin : Real.sin α = 2 * Real.sqrt 2 / 3) (hcos : Real.cos α = 1 / 3) :
  Real.tan α = 2 * Real.sqrt 2 :=
sorry

theorem f_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) :
  f α = -2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_f_alpha_l2342_234297


namespace NUMINAMATH_GPT_value_of_a5_l2342_234236

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a n * r ^ (m - n) = a m

theorem value_of_a5 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a5_l2342_234236


namespace NUMINAMATH_GPT_jerry_bought_3_pounds_l2342_234299

-- Definitions based on conditions:
def cost_mustard_oil := 2 * 13
def cost_pasta_sauce := 5
def total_money := 50
def money_left := 7
def cost_gluten_free_pasta_per_pound := 4

-- The proof goal based on the correct answer:
def pounds_gluten_free_pasta : Nat :=
  let total_spent := total_money - money_left
  let spent_on_mustard_and_sauce := cost_mustard_oil + cost_pasta_sauce
  let spent_on_pasta := total_spent - spent_on_mustard_and_sauce
  spent_on_pasta / cost_gluten_free_pasta_per_pound

theorem jerry_bought_3_pounds :
  pounds_gluten_free_pasta = 3 := by
  -- the proof should follow here
  sorry

end NUMINAMATH_GPT_jerry_bought_3_pounds_l2342_234299


namespace NUMINAMATH_GPT_interest_rate_same_l2342_234244

theorem interest_rate_same (initial_amount: ℝ) (interest_earned: ℝ) 
  (time_period1: ℝ) (time_period2: ℝ) (principal: ℝ) (initial_rate: ℝ) : 
  initial_amount * initial_rate * time_period2 = interest_earned * 100 ↔ initial_rate = 12 
  :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_same_l2342_234244


namespace NUMINAMATH_GPT_circle_center_l2342_234295

theorem circle_center (n : ℝ) (r : ℝ) (h1 : r = 7) (h2 : ∀ x : ℝ, x^2 + (x^2 - n)^2 = 49 → x^4 - x^2 * (2*n - 1) + n^2 - 49 = 0)
  (h3 : ∃! y : ℝ, y^2 + (1 - 2*n) * y + n^2 - 49 = 0) :
  (0, n) = (0, 197 / 4) := 
sorry

end NUMINAMATH_GPT_circle_center_l2342_234295


namespace NUMINAMATH_GPT_total_weight_is_correct_l2342_234208

-- Define the weight of apples
def weight_of_apples : ℕ := 240

-- Define the multiplier for pears
def pears_multiplier : ℕ := 3

-- Define the weight of pears
def weight_of_pears := pears_multiplier * weight_of_apples

-- Define the total weight of apples and pears
def total_weight : ℕ := weight_of_apples + weight_of_pears

-- The theorem that states the total weight calculation
theorem total_weight_is_correct : total_weight = 960 := by
  sorry

end NUMINAMATH_GPT_total_weight_is_correct_l2342_234208


namespace NUMINAMATH_GPT_solve_problem_l2342_234205

-- Define the variables and conditions
def problem_statement : Prop :=
  ∃ x : ℕ, 865 * 48 = 240 * x ∧ x = 173

-- Statement to prove
theorem solve_problem : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l2342_234205


namespace NUMINAMATH_GPT_window_width_correct_l2342_234200

def total_width_window (x : ℝ) : ℝ :=
  let pane_width := 4 * x
  let num_panes_per_row := 4
  let num_borders := 5
  num_panes_per_row * pane_width + num_borders * 3

theorem window_width_correct (x : ℝ) :
  total_width_window x = 16 * x + 15 := sorry

end NUMINAMATH_GPT_window_width_correct_l2342_234200


namespace NUMINAMATH_GPT_construct_trihedral_angle_l2342_234277

-- Define the magnitudes of dihedral angles
variables (α β γ : ℝ)

-- Problem statement
theorem construct_trihedral_angle (h₀ : 0 < α) (h₁ : 0 < β) (h₂ : 0 < γ) :
  ∃ (trihedral_angle : Type), true := 
sorry

end NUMINAMATH_GPT_construct_trihedral_angle_l2342_234277


namespace NUMINAMATH_GPT_inequality_of_four_numbers_l2342_234257

theorem inequality_of_four_numbers 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a ≤ 3 * b) (h2 : b ≤ 3 * a) (h3 : a ≤ 3 * c)
  (h4 : c ≤ 3 * a) (h5 : a ≤ 3 * d) (h6 : d ≤ 3 * a)
  (h7 : b ≤ 3 * c) (h8 : c ≤ 3 * b) (h9 : b ≤ 3 * d)
  (h10 : d ≤ 3 * b) (h11 : c ≤ 3 * d) (h12 : d ≤ 3 * c) : 
  a^2 + b^2 + c^2 + d^2 < 2 * (ab + ac + ad + bc + bd + cd) :=
sorry

end NUMINAMATH_GPT_inequality_of_four_numbers_l2342_234257


namespace NUMINAMATH_GPT_angle_bisector_slope_l2342_234224

theorem angle_bisector_slope (k : ℚ) : 
  (∀ x : ℚ, (y = 2 * x ∧ y = 4 * x) → (y = k * x)) → k = -12 / 7 :=
sorry

end NUMINAMATH_GPT_angle_bisector_slope_l2342_234224


namespace NUMINAMATH_GPT_fraction_not_integer_l2342_234228

theorem fraction_not_integer (a b : ℤ) : ¬ (∃ k : ℤ, (a^2 + b^2) = k * (a^2 - b^2)) :=
sorry

end NUMINAMATH_GPT_fraction_not_integer_l2342_234228


namespace NUMINAMATH_GPT_percentage_exceed_l2342_234248

theorem percentage_exceed (x y : ℝ) (h : y = x + 0.2 * x) :
  (y - x) / x * 100 = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_exceed_l2342_234248


namespace NUMINAMATH_GPT_area_of_grey_region_l2342_234231

open Nat

theorem area_of_grey_region
  (a1 a2 b : ℕ)
  (h1 : a1 = 8 * 10)
  (h2 : a2 = 9 * 12)
  (hb : b = 37)
  : (a2 - (a1 - b) = 65) := by
  sorry

end NUMINAMATH_GPT_area_of_grey_region_l2342_234231


namespace NUMINAMATH_GPT_min_value_frac_ineq_l2342_234245

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : a + b = 5) : 
  (1 / (a - 1) + 9 / (b - 2)) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_frac_ineq_l2342_234245


namespace NUMINAMATH_GPT_trapezoid_perimeter_l2342_234241

noncomputable def isosceles_trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) : ℝ :=
  8 * R / (Real.sin α)

theorem trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) :
  ∃ (P : ℝ), P = isosceles_trapezoid_perimeter R α hα := by
    sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l2342_234241


namespace NUMINAMATH_GPT_faster_speed_l2342_234238

theorem faster_speed (D : ℝ) (v : ℝ) (h₁ : D = 33.333333333333336) 
                      (h₂ : 10 * (D + 20) = v * D) : v = 16 :=
by
  sorry

end NUMINAMATH_GPT_faster_speed_l2342_234238


namespace NUMINAMATH_GPT_cos_value_l2342_234249

theorem cos_value (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (2 * π / 3 - α) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_value_l2342_234249


namespace NUMINAMATH_GPT_max_trees_l2342_234243

theorem max_trees (interval distance road_length number_of_intervals add_one : ℕ) 
  (h_interval: interval = 4) 
  (h_distance: distance = 28) 
  (h_intervals: number_of_intervals = distance / interval)
  (h_add: add_one = number_of_intervals + 1) :
  add_one = 8 :=
sorry

end NUMINAMATH_GPT_max_trees_l2342_234243


namespace NUMINAMATH_GPT_ab_value_l2342_234291

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 172) : ab = 85 / 6 := 
by
  sorry

end NUMINAMATH_GPT_ab_value_l2342_234291


namespace NUMINAMATH_GPT_Canada_moose_population_l2342_234214

theorem Canada_moose_population (moose beavers humans : ℕ) (h1 : beavers = 2 * moose) 
                              (h2 : humans = 19 * beavers) (h3 : humans = 38 * 10^6) : 
                              moose = 1 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_Canada_moose_population_l2342_234214


namespace NUMINAMATH_GPT_smaller_number_l2342_234294

theorem smaller_number (x y : ℝ) (h1 : y - x = (1 / 3) * y) (h2 : y = 71.99999999999999) : x = 48 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_l2342_234294


namespace NUMINAMATH_GPT_probability_of_point_in_smaller_square_l2342_234232

-- Definitions
def A_large : ℝ := 5 * 5
def A_small : ℝ := 2 * 2

-- Theorem statement
theorem probability_of_point_in_smaller_square 
  (side_large : ℝ) (side_small : ℝ)
  (hle : side_large = 5) (hse : side_small = 2) :
  (side_large * side_large ≠ 0) ∧ (side_small * side_small ≠ 0) → 
  (A_small / A_large = 4 / 25) :=
sorry

end NUMINAMATH_GPT_probability_of_point_in_smaller_square_l2342_234232


namespace NUMINAMATH_GPT_daniel_earnings_l2342_234211

theorem daniel_earnings :
  let monday_fabric := 20
  let monday_yarn := 15
  let tuesday_fabric := 2 * monday_fabric
  let tuesday_yarn := monday_yarn + 10
  let wednesday_fabric := (1 / 4) * tuesday_fabric
  let wednesday_yarn := (1 / 2) * tuesday_yarn
  let total_fabric := monday_fabric + tuesday_fabric + wednesday_fabric
  let total_yarn := monday_yarn + tuesday_yarn + wednesday_yarn
  let fabric_cost := 2
  let yarn_cost := 3
  let fabric_earnings_before_discount := total_fabric * fabric_cost
  let yarn_earnings_before_discount := total_yarn * yarn_cost
  let fabric_discount := if total_fabric > 30 then 0.10 * fabric_earnings_before_discount else 0
  let yarn_discount := if total_yarn > 20 then 0.05 * yarn_earnings_before_discount else 0
  let fabric_earnings_after_discount := fabric_earnings_before_discount - fabric_discount
  let yarn_earnings_after_discount := yarn_earnings_before_discount - yarn_discount
  let total_earnings := fabric_earnings_after_discount + yarn_earnings_after_discount
  total_earnings = 275.625 := by
  {
    sorry
  }

end NUMINAMATH_GPT_daniel_earnings_l2342_234211


namespace NUMINAMATH_GPT_find_m_l2342_234220

def A (m : ℤ) : Set ℤ := {2, 5, m ^ 2 - m}
def B (m : ℤ) : Set ℤ := {2, m + 3}

theorem find_m (m : ℤ) : A m ∩ B m = B m → m = 3 := by
  sorry

end NUMINAMATH_GPT_find_m_l2342_234220


namespace NUMINAMATH_GPT_triangle_base_length_l2342_234213

/-
Theorem: Given a triangle with height 5.8 meters and area 24.36 square meters,
the length of the base is 8.4 meters.
-/

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) :
  h = 5.8 ∧ A = 24.36 ∧ A = (b * h) / 2 → b = 8.4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_length_l2342_234213


namespace NUMINAMATH_GPT_hotel_elevator_cubic_value_l2342_234233

noncomputable def hotel_elevator_cubic : ℚ → ℚ := sorry

theorem hotel_elevator_cubic_value :
  hotel_elevator_cubic 11 = 11 ∧
  hotel_elevator_cubic 12 = 12 ∧
  hotel_elevator_cubic 13 = 14 ∧
  hotel_elevator_cubic 14 = 15 →
  hotel_elevator_cubic 15 = 13 :=
sorry

end NUMINAMATH_GPT_hotel_elevator_cubic_value_l2342_234233


namespace NUMINAMATH_GPT_f_g_5_eq_163_l2342_234283

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem f_g_5_eq_163 : f (g 5) = 163 := by
  sorry

end NUMINAMATH_GPT_f_g_5_eq_163_l2342_234283


namespace NUMINAMATH_GPT_london_to_baglmintster_distance_l2342_234253

variable (D : ℕ) -- distance from London to Baglmintster

-- Conditions
def meeting_point_condition_1 := D ≥ 40
def meeting_point_condition_2 := D ≥ 48
def initial_meeting := D - 40
def return_meeting := D - 48

theorem london_to_baglmintster_distance :
  (D - 40) + 48 = D + 8 ∧ 40 + (D - 48) = D - 8 → D = 72 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_london_to_baglmintster_distance_l2342_234253


namespace NUMINAMATH_GPT_unique_prime_with_conditions_l2342_234215

theorem unique_prime_with_conditions (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (p + 2)) (hp4 : Nat.Prime (p + 4)) : p = 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_prime_with_conditions_l2342_234215


namespace NUMINAMATH_GPT_find_c_l2342_234293

theorem find_c (c : ℝ) (h : ∃ a : ℝ, x^2 - 50 * x + c = (x - a)^2) : c = 625 :=
  by
  sorry

end NUMINAMATH_GPT_find_c_l2342_234293


namespace NUMINAMATH_GPT_leak_time_to_empty_l2342_234206

def pump_rate : ℝ := 0.1 -- P = 0.1 tanks/hour
def effective_rate : ℝ := 0.05 -- P - L = 0.05 tanks/hour

theorem leak_time_to_empty (P L : ℝ) (hp : P = pump_rate) (he : P - L = effective_rate) :
  1 / L = 20 := by
  sorry

end NUMINAMATH_GPT_leak_time_to_empty_l2342_234206


namespace NUMINAMATH_GPT_smallest_value_among_options_l2342_234276

theorem smallest_value_among_options (x : ℕ) (h : x = 9) :
    min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min ((x+3)/8) ((x-3)/8)))) = (3/4) :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_among_options_l2342_234276


namespace NUMINAMATH_GPT_max_q_minus_r_839_l2342_234234

theorem max_q_minus_r_839 : ∃ (q r : ℕ), (839 = 19 * q + r) ∧ (0 ≤ r ∧ r < 19) ∧ q - r = 41 :=
by
  sorry

end NUMINAMATH_GPT_max_q_minus_r_839_l2342_234234


namespace NUMINAMATH_GPT_combined_weight_l2342_234202

-- Given constants
def JakeWeight : ℕ := 198
def WeightLost : ℕ := 8
def KendraWeight := (JakeWeight - WeightLost) / 2

-- Prove the combined weight of Jake and Kendra
theorem combined_weight : JakeWeight + KendraWeight = 293 := by
  sorry

end NUMINAMATH_GPT_combined_weight_l2342_234202


namespace NUMINAMATH_GPT_A_minus_B_l2342_234221

theorem A_minus_B (x y m n A B : ℤ) (hx : x > y) (hx1 : x + y = 7) (hx2 : x * y = 12)
                  (hm : m > n) (hm1 : m + n = 13) (hm2 : m^2 + n^2 = 97)
                  (hA : A = x - y) (hB : B = m - n) :
                  A - B = -4 := by
  sorry

end NUMINAMATH_GPT_A_minus_B_l2342_234221


namespace NUMINAMATH_GPT_base7_digits_of_143_l2342_234286

theorem base7_digits_of_143 : ∃ d1 d2 d3 : ℕ, (d1 < 7 ∧ d2 < 7 ∧ d3 < 7) ∧ (143 = d1 * 49 + d2 * 7 + d3) ∧ (d1 = 2 ∧ d2 = 6 ∧ d3 = 3) :=
by
  sorry

end NUMINAMATH_GPT_base7_digits_of_143_l2342_234286


namespace NUMINAMATH_GPT_cars_meet_after_40_minutes_l2342_234203

noncomputable def time_to_meet 
  (BC CD : ℝ) (speed : ℝ) 
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) : ℝ :=
  (BC + CD) / speed * 40 / 60

-- Define the condition that must hold: cars meet at 40 minutes
theorem cars_meet_after_40_minutes
  (BC CD : ℝ) (speed : ℝ)
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) :
  time_to_meet BC CD speed constant_speed = 40 := sorry

end NUMINAMATH_GPT_cars_meet_after_40_minutes_l2342_234203


namespace NUMINAMATH_GPT_factor_correct_l2342_234252

theorem factor_correct (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_GPT_factor_correct_l2342_234252


namespace NUMINAMATH_GPT_distance_greater_than_two_l2342_234296

theorem distance_greater_than_two (x : ℝ) (h : |x| > 2) : x > 2 ∨ x < -2 :=
sorry

end NUMINAMATH_GPT_distance_greater_than_two_l2342_234296


namespace NUMINAMATH_GPT_triangle_height_l2342_234260

theorem triangle_height (base area height : ℝ)
    (h_base : base = 4)
    (h_area : area = 16)
    (h_area_formula : area = (base * height) / 2) :
    height = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l2342_234260


namespace NUMINAMATH_GPT_lisa_needs_additional_marbles_l2342_234240

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end NUMINAMATH_GPT_lisa_needs_additional_marbles_l2342_234240


namespace NUMINAMATH_GPT_fraction_of_menu_items_i_can_eat_l2342_234264

def total_dishes (vegan_dishes non_vegan_dishes : ℕ) : ℕ := vegan_dishes + non_vegan_dishes

def vegan_dishes_without_soy (vegan_dishes vegan_with_soy : ℕ) : ℕ := vegan_dishes - vegan_with_soy

theorem fraction_of_menu_items_i_can_eat (vegan_dishes non_vegan_dishes vegan_with_soy : ℕ)
  (h_vegan_dishes : vegan_dishes = 6)
  (h_menu_total : vegan_dishes = (total_dishes vegan_dishes non_vegan_dishes) / 3)
  (h_vegan_with_soy : vegan_with_soy = 4)
  : (vegan_dishes_without_soy vegan_dishes vegan_with_soy) / (total_dishes vegan_dishes non_vegan_dishes) = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_menu_items_i_can_eat_l2342_234264


namespace NUMINAMATH_GPT_cat_food_customers_l2342_234280

/-
Problem: There was a big sale on cat food at the pet store. Some people bought cat food that day. The first 8 customers bought 3 cases each. The next four customers bought 2 cases each. The last 8 customers of the day only bought 1 case each. In total, 40 cases of cat food were sold. How many people bought cat food that day?
-/

theorem cat_food_customers:
  (8 * 3) + (4 * 2) + (8 * 1) = 40 →
  8 + 4 + 8 = 20 :=
by
  intro h
  linarith

end NUMINAMATH_GPT_cat_food_customers_l2342_234280


namespace NUMINAMATH_GPT_negation_of_at_most_four_l2342_234298

theorem negation_of_at_most_four (n : ℕ) : ¬(n ≤ 4) → n ≥ 5 := 
by
  sorry

end NUMINAMATH_GPT_negation_of_at_most_four_l2342_234298


namespace NUMINAMATH_GPT_find_a_l2342_234272

theorem find_a (r s a : ℚ) (h1 : s^2 = 16) (h2 : 2 * r * s = 15) (h3 : a = r^2) : a = 225/64 := by
  sorry

end NUMINAMATH_GPT_find_a_l2342_234272


namespace NUMINAMATH_GPT_smallest_x_for_multiple_l2342_234261

theorem smallest_x_for_multiple (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 640 = 2^7 * 5^1) :
  (450 * x) % 640 = 0 ↔ x = 64 :=
sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_l2342_234261


namespace NUMINAMATH_GPT_solve_inequality_l2342_234227

def inequality_solution (x : ℝ) : Prop := |2 * x - 1| - x ≥ 2 

theorem solve_inequality (x : ℝ) : 
  inequality_solution x ↔ (x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l2342_234227


namespace NUMINAMATH_GPT_solve_for_x_l2342_234275

theorem solve_for_x (y : ℝ) (x : ℝ) 
  (h : x / (x - 1) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 3)) : 
  x = (y^2 + 3 * y - 2) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l2342_234275


namespace NUMINAMATH_GPT_ratio_of_hypothetical_to_actual_children_l2342_234218

theorem ratio_of_hypothetical_to_actual_children (C H : ℕ) 
  (h1 : H = 16 * 8)
  (h2 : ∀ N : ℕ, N = C / 8 → C * N = 512) 
  (h3 : C^2 = 512 * 8) : H / C = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_hypothetical_to_actual_children_l2342_234218


namespace NUMINAMATH_GPT_log_bound_sum_l2342_234219

theorem log_bound_sum (c d : ℕ) (h_c : c = 10) (h_d : d = 11) (h_bound : 10 < Real.log 1350 / Real.log 2 ∧ Real.log 1350 / Real.log 2 < 11) : c + d = 21 :=
by
  -- omitted proof
  sorry

end NUMINAMATH_GPT_log_bound_sum_l2342_234219


namespace NUMINAMATH_GPT_quadratic_eq_one_solution_m_eq_49_div_12_l2342_234262

theorem quadratic_eq_one_solution_m_eq_49_div_12 (m : ℝ) : 
  (∃ m, ∀ x, 3 * x ^ 2 - 7 * x + m = 0 → (b^2 - 4 * a * c = 0) → m = 49 / 12) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_one_solution_m_eq_49_div_12_l2342_234262


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l2342_234201

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let a := 1
  let b := 2*k - 1
  let c := -k - 1
  discriminant a b c > 0 := by
  sorry

theorem determine_k_from_roots_relation (x1 x2 k : ℝ) 
  (h1 : x1 + x2 = -(2*k - 1))
  (h2 : x1 * x2 = -k - 1)
  (h3 : x1 + x2 - 4*(x1 * x2) = 2) :
  k = -3/2 := by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_determine_k_from_roots_relation_l2342_234201


namespace NUMINAMATH_GPT_log_product_computation_l2342_234269

theorem log_product_computation : 
  (Real.log 32 / Real.log 2) * (Real.log 27 / Real.log 3) = 15 := 
by
  -- The proof content, which will be skipped with 'sorry'.
  sorry

end NUMINAMATH_GPT_log_product_computation_l2342_234269


namespace NUMINAMATH_GPT_average_visitors_per_day_l2342_234251

/-- A library has different visitor numbers depending on the day of the week.
  - On Sundays, the library has an average of 660 visitors.
  - On Mondays through Thursdays, there are 280 visitors on average.
  - Fridays and Saturdays see an increase to an average of 350 visitors.
  - This month has a special event on the third Saturday, bringing an extra 120 visitors that day.
  - The month has 30 days and begins with a Sunday.
  We want to calculate the average number of visitors per day for the entire month. -/
theorem average_visitors_per_day
  (num_days : ℕ) (starts_on_sunday : Bool)
  (sundays_visitors : ℕ) (weekdays_visitors : ℕ) (weekend_visitors : ℕ)
  (special_event_extra_visitors : ℕ) (sundays : ℕ) (mondays : ℕ)
  (tuesdays : ℕ) (wednesdays : ℕ) (thursdays : ℕ) (fridays : ℕ)
  (saturdays : ℕ) :
  num_days = 30 → starts_on_sunday = true →
  sundays_visitors = 660 → weekdays_visitors = 280 → weekend_visitors = 350 →
  special_event_extra_visitors = 120 →
  sundays = 4 → mondays = 5 →
  tuesdays = 4 → wednesdays = 4 → thursdays = 4 → fridays = 4 → saturdays = 4 →
  ((sundays * sundays_visitors +
    mondays * weekdays_visitors +
    tuesdays * weekdays_visitors +
    wednesdays * weekdays_visitors +
    thursdays * weekdays_visitors +
    fridays * weekend_visitors +
    saturdays * weekend_visitors +
    special_event_extra_visitors) / num_days = 344) :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_l2342_234251


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_3_and_8_l2342_234263

theorem smallest_four_digit_divisible_by_3_and_8 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 3 = 0 ∧ n % 8 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 3 = 0 ∧ m % 8 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_3_and_8_l2342_234263
