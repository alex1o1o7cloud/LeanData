import Mathlib

namespace drowning_ratio_l842_84264

variable (total_sheep total_cows total_dogs drowned_sheep drowned_cows total_animals : ℕ)

-- Conditions provided
variable (initial_conditions : total_sheep = 20 ∧ total_cows = 10 ∧ total_dogs = 14)
variable (sheep_drowned_condition : drowned_sheep = 3)
variable (dogs_shore_condition : total_dogs = 14)
variable (total_made_it_shore : total_animals = 35)

theorem drowning_ratio (h1 : total_sheep = 20) (h2 : total_cows = 10) (h3 : total_dogs = 14) 
    (h4 : drowned_sheep = 3) (h5 : total_animals = 35) 
    : (drowned_cows = 2 * drowned_sheep) :=
by
  sorry

end drowning_ratio_l842_84264


namespace grandma_gave_each_l842_84256

-- Define the conditions
def gasoline: ℝ := 8
def lunch: ℝ := 15.65
def gifts: ℝ := 5 * 2  -- $5 each for two persons
def total_spent: ℝ := gasoline + lunch + gifts
def initial_amount: ℝ := 50
def amount_left: ℝ := 36.35

-- Define the proof problem
theorem grandma_gave_each :
  (amount_left - (initial_amount - total_spent)) / 2 = 10 :=
by
  sorry

end grandma_gave_each_l842_84256


namespace transform_fraction_l842_84240

theorem transform_fraction (x : ℝ) (h : x ≠ 1) : - (1 / (1 - x)) = 1 / (x - 1) :=
by
  sorry

end transform_fraction_l842_84240


namespace family_members_count_l842_84239

-- Defining the conditions given in the problem
variables (cyrus_bites_arms_legs : ℕ) (cyrus_bites_body : ℕ) (total_bites_family : ℕ)
variables (family_bites_per_person : ℕ) (cyrus_total_bites : ℕ)

-- Given conditions
def condition1 : cyrus_bites_arms_legs = 14 := sorry
def condition2 : cyrus_bites_body = 10 := sorry
def condition3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body := sorry
def condition4 : total_bites_family = cyrus_total_bites / 2 := sorry
def condition5 : ∀ n : ℕ, total_bites_family = n * family_bites_per_person := sorry

-- The theorem to prove: The number of people in the rest of Cyrus' family is 12
theorem family_members_count (n : ℕ) (h1 : cyrus_bites_arms_legs = 14)
    (h2 : cyrus_bites_body = 10) (h3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body)
    (h4 : total_bites_family = cyrus_total_bites / 2)
    (h5 : ∀ n, total_bites_family = n * family_bites_per_person) : n = 12 :=
sorry

end family_members_count_l842_84239


namespace num_ways_two_different_colors_l842_84259

theorem num_ways_two_different_colors 
  (red white blue : ℕ) 
  (total_balls : ℕ) 
  (choose : ℕ → ℕ → ℕ) 
  (h_red : red = 2) 
  (h_white : white = 3) 
  (h_blue : blue = 1) 
  (h_total : total_balls = red + white + blue) 
  (h_choose_total : choose total_balls 3 = 20)
  (h_choose_three_diff_colors : 2 * 3 * 1 = 6)
  (h_one_color : 1 = 1) :
  choose total_balls 3 - 6 - 1 = 13 := 
by
  sorry

end num_ways_two_different_colors_l842_84259


namespace treasures_on_island_l842_84276

-- Define the propositions P and K
def P : Prop := ∃ p : Prop, p
def K : Prop := ∃ k : Prop, k

-- Define the claim by A
def A_claim : Prop := K ↔ P

-- Theorem statement as specified part (b)
theorem treasures_on_island (A_is_knight_or_liar : (A_claim ↔ true) ∨ (A_claim ↔ false)) : ∃ P, P :=
by
  sorry

end treasures_on_island_l842_84276


namespace number_of_pipes_l842_84269

theorem number_of_pipes (d_large d_small: ℝ) (π : ℝ) (h1: d_large = 4) (h2: d_small = 2) : 
  ((π * (d_large / 2)^2) / (π * (d_small / 2)^2) = 4) := 
by
  sorry

end number_of_pipes_l842_84269


namespace isosceles_triangle_perimeter_l842_84218

theorem isosceles_triangle_perimeter :
  ∀ x y : ℝ, x^2 - 7*x + 10 = 0 → y^2 - 7*y + 10 = 0 → x ≠ y → x + x + y = 12 :=
by
  intros x y hx hy hxy
  -- Place for proof
  sorry

end isosceles_triangle_perimeter_l842_84218


namespace customers_not_tipping_l842_84280

theorem customers_not_tipping (number_of_customers tip_per_customer total_earned_in_tips : ℕ)
  (h_number : number_of_customers = 7)
  (h_tip : tip_per_customer = 3)
  (h_earned : total_earned_in_tips = 6) :
  number_of_customers - (total_earned_in_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_not_tipping_l842_84280


namespace club_co_presidents_l842_84257

def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem club_co_presidents : choose 18 3 = 816 := by
  sorry

end club_co_presidents_l842_84257


namespace charley_pencils_final_count_l842_84213

def charley_initial_pencils := 50
def lost_pencils_while_moving := 8
def misplaced_fraction_first_week := 1 / 3
def lost_fraction_second_week := 1 / 4

theorem charley_pencils_final_count:
  let initial := charley_initial_pencils
  let after_moving := initial - lost_pencils_while_moving
  let misplaced_first_week := misplaced_fraction_first_week * after_moving
  let remaining_after_first_week := after_moving - misplaced_first_week
  let lost_second_week := lost_fraction_second_week * remaining_after_first_week
  let final_pencils := remaining_after_first_week - lost_second_week
  final_pencils = 21 := 
sorry

end charley_pencils_final_count_l842_84213


namespace ratio_of_x_to_y_l842_84203

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 3 / 5) : x / y = 16 / 15 :=
sorry

end ratio_of_x_to_y_l842_84203


namespace grapes_purchased_l842_84279

-- Define the given conditions
def price_per_kg_grapes : ℕ := 68
def kg_mangoes : ℕ := 9
def price_per_kg_mangoes : ℕ := 48
def total_paid : ℕ := 908

-- Define the proof problem
theorem grapes_purchased : ∃ (G : ℕ), (price_per_kg_grapes * G + price_per_kg_mangoes * kg_mangoes = total_paid) ∧ (G = 7) :=
by {
  use 7,
  sorry
}

end grapes_purchased_l842_84279


namespace steel_parts_count_l842_84261

-- Definitions for conditions
variables (a b : ℕ)

-- Conditions provided in the problem
axiom machines_count : a + b = 21
axiom chrome_parts : 2 * a + 4 * b = 66

-- Statement to prove
theorem steel_parts_count : 3 * a + 2 * b = 51 :=
by
  sorry

end steel_parts_count_l842_84261


namespace min_value_inequality_l842_84206

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z ≥ 36) ∧ 
  ((1 / x + 4 / y + 9 / z = 36) ↔ (x = 1 / 6 ∧ y = 1 / 3 ∧ z = 1 / 2)) :=
by
  sorry

end min_value_inequality_l842_84206


namespace pen_cost_l842_84216

variable (p i : ℝ)

theorem pen_cost (h1 : p + i = 1.10) (h2 : p = 1 + i) : p = 1.05 :=
by 
  -- proof steps here
  sorry

end pen_cost_l842_84216


namespace cos_A_value_l842_84207

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
-- a, b, c are the sides opposite to angles A, B, and C respectively.
-- Assumption 1: b - c = (1/4) * a
def condition1 := b - c = (1/4) * a
-- Assumption 2: 2 * sin B = 3 * sin C
def condition2 := 2 * Real.sin B = 3 * Real.sin C

-- The theorem statement: Under these conditions, prove that cos A = -1/4.
theorem cos_A_value (h1 : condition1 a b c) (h2 : condition2 B C) : 
    Real.cos A = -1/4 :=
sorry -- placeholder for the proof

end cos_A_value_l842_84207


namespace find_real_solutions_l842_84228

noncomputable def cubic_eq_solutions (x : ℝ) : Prop := 
  x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3

theorem find_real_solutions : {x : ℝ | cubic_eq_solutions x} = {6} :=
by
  sorry

end find_real_solutions_l842_84228


namespace length_of_hypotenuse_l842_84246

/-- Define the problem's parameters -/
def perimeter : ℝ := 34
def area : ℝ := 24
def length_hypotenuse (a b c : ℝ) : Prop := a + b + c = perimeter 
  ∧ (1/2) * a * b = area
  ∧ a^2 + b^2 = c^2

/- Lean statement for the proof problem -/
theorem length_of_hypotenuse (a b c : ℝ) 
  (h1: a + b + c = 34)
  (h2: (1/2) * a * b = 24)
  (h3: a^2 + b^2 = c^2)
  : c = 62 / 4 := sorry

end length_of_hypotenuse_l842_84246


namespace tree_heights_l842_84254

theorem tree_heights :
  let Tree_A := 150
  let Tree_B := (2/3 : ℝ) * Tree_A
  let Tree_C := (1/2 : ℝ) * Tree_B
  let Tree_D := Tree_C + 25
  let Tree_E := 0.40 * Tree_A
  let Tree_F := (Tree_B + Tree_D) / 2
  let Tree_G := (3/8 : ℝ) * Tree_A
  let Tree_H := 1.25 * Tree_F
  let Tree_I := 0.60 * (Tree_E + Tree_G)
  let total_height := Tree_A + Tree_B + Tree_C + Tree_D + Tree_E + Tree_F + Tree_G + Tree_H + Tree_I
  Tree_A = 150 ∧
  Tree_B = 100 ∧
  Tree_C = 50 ∧
  Tree_D = 75 ∧
  Tree_E = 60 ∧
  Tree_F = 87.5 ∧
  Tree_G = 56.25 ∧
  Tree_H = 109.375 ∧
  Tree_I = 69.75 ∧
  total_height = 758.125 :=
by
  sorry

end tree_heights_l842_84254


namespace conic_sections_with_foci_at_F2_zero_l842_84258

theorem conic_sections_with_foci_at_F2_zero (a b m n: ℝ) (h1 : a > b) (h2: b > 0) (h3: m > 0) (h4: n > 0) (h5: a^2 - b^2 = 4) (h6: m^2 + n^2 = 4):
  (∀ x y: ℝ, x^2 / (a^2) + y^2 / (b^2) = 1) ∧ (∀ x y: ℝ, x^2 / (11/60) + y^2 / (11/16) = 1) ∧ 
  ∀ x y: ℝ, x^2 / (m^2) - y^2 / (n^2) = 1 ∧ ∀ x y: ℝ, 5*x^2 / 4 - 5*y^2 / 16 = 1 := 
sorry

end conic_sections_with_foci_at_F2_zero_l842_84258


namespace maria_average_speed_l842_84289

noncomputable def average_speed (total_distance : ℕ) (total_time : ℕ) : ℚ :=
  total_distance / total_time

theorem maria_average_speed :
  average_speed 200 7 = 28 + 4 / 7 :=
sorry

end maria_average_speed_l842_84289


namespace fraction_spent_first_week_l842_84215

theorem fraction_spent_first_week
  (S : ℝ) (F : ℝ)
  (h1 : S > 0)
  (h2 : F * S + 3 * (0.20 * S) + 0.15 * S = S) : 
  F = 0.25 := 
sorry

end fraction_spent_first_week_l842_84215


namespace find_a6_l842_84285

noncomputable def a_n (n : ℕ) : ℝ := sorry
noncomputable def S_n (n : ℕ) : ℝ := sorry
noncomputable def r : ℝ := sorry

axiom h_pos : ∀ n, a_n n > 0
axiom h_s3 : S_n 3 = 14
axiom h_a3 : a_n 3 = 8

theorem find_a6 : a_n 6 = 64 := by sorry

end find_a6_l842_84285


namespace episodes_relationship_l842_84278

variable (x y z : ℕ)

theorem episodes_relationship 
  (h1 : x * z = 50) 
  (h2 : y * z = 75) : 
  y = (3 / 2) * x ∧ z = 50 / x := 
by
  sorry

end episodes_relationship_l842_84278


namespace greatest_five_digit_number_sum_of_digits_l842_84217

def is_five_digit_number (n : ℕ) : Prop :=
  10000 <= n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * (n / 10000)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + (n / 10000)

theorem greatest_five_digit_number_sum_of_digits (M : ℕ) 
  (h1 : is_five_digit_number M) 
  (h2 : digits_product M = 210) :
  digits_sum M = 20 := 
sorry

end greatest_five_digit_number_sum_of_digits_l842_84217


namespace domain_change_l842_84273

theorem domain_change (f : ℝ → ℝ) :
  (∀ x : ℝ, -2 ≤ x + 1 ∧ x + 1 ≤ 3) →
  (∀ x : ℝ, -2 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 3) →
  ∀ x : ℝ, -3 / 2 ≤ x ∧ x ≤ 1 :=
by {
  sorry
}

end domain_change_l842_84273


namespace R2_area_is_160_l842_84202

-- Define the initial conditions.
structure Rectangle :=
(width : ℝ)
(height : ℝ)

def R1 : Rectangle := { width := 4, height := 8 }

def similar (r1 r2 : Rectangle) : Prop :=
  r2.width / r2.height = r1.width / r1.height

def R2_diagonal := 20

-- Proving that the area of R2 is 160 square inches
theorem R2_area_is_160 (R2 : Rectangle)
  (h_similar : similar R1 R2)
  (h_diagonal : R2.width^2 + R2.height^2 = R2_diagonal^2) :
  R2.width * R2.height = 160 :=
  sorry

end R2_area_is_160_l842_84202


namespace relationship_among_a_b_c_l842_84250

theorem relationship_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (4 : ℝ) ^ (1 / 2))
  (hb : b = (2 : ℝ) ^ (1 / 3))
  (hc : c = (5 : ℝ) ^ (1 / 2))
: b < a ∧ a < c := 
sorry

end relationship_among_a_b_c_l842_84250


namespace garden_fencing_l842_84219

theorem garden_fencing (length width : ℕ) (h1 : length = 80) (h2 : width = length / 2) : 2 * (length + width) = 240 :=
by
  sorry

end garden_fencing_l842_84219


namespace ratio_of_birds_to_trees_and_stones_l842_84224

theorem ratio_of_birds_to_trees_and_stones (stones birds : ℕ) (h_stones : stones = 40)
  (h_birds : birds = 400) (trees : ℕ) (h_trees : trees = 3 * stones + stones) :
  (birds : ℚ) / (trees + stones) = 2 :=
by
  -- The actual proof steps would go here.
  sorry

end ratio_of_birds_to_trees_and_stones_l842_84224


namespace daughter_age_l842_84204

theorem daughter_age (m d : ℕ) (h1 : m + d = 60) (h2 : m - 10 = 7 * (d - 10)) : d = 15 :=
sorry

end daughter_age_l842_84204


namespace third_candidate_votes_l842_84231

theorem third_candidate_votes (V A B W: ℕ) (hA : A = 2500) (hB : B = 15000) 
  (hW : W = (2 * V) / 3) (hV : V = W + A + B) : (V - (A + B)) = 35000 := by
  sorry

end third_candidate_votes_l842_84231


namespace students_answered_both_correctly_l842_84238

theorem students_answered_both_correctly 
  (total_students : ℕ) (took_test : ℕ) 
  (q1_correct : ℕ) (q2_correct : ℕ)
  (did_not_take_test : ℕ)
  (h1 : total_students = 25)
  (h2 : q1_correct = 22)
  (h3 : q2_correct = 20)
  (h4 : did_not_take_test = 3)
  (h5 : took_test = total_students - did_not_take_test) :
  (q1_correct + q2_correct) - took_test = 20 := 
by 
  -- Proof skipped.
  sorry

end students_answered_both_correctly_l842_84238


namespace sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l842_84286

noncomputable def y (x m : ℝ) : ℝ := x^2 + m / x
noncomputable def y_prime (x m : ℝ) : ℝ := 2 * x - m / x^2

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x ≥ 1, y_prime x m ≥ 0) ↔ m ≤ 2 :=
sorry  -- Proof skipped as instructed

-- Now, state that m < 1 is a sufficient but not necessary condition
theorem m_sufficient_but_not_necessary (m : ℝ) :
  m < 1 → (∀ x ≥ 1, y_prime x m ≥ 0) :=
sorry  -- Proof skipped as instructed

end sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l842_84286


namespace algebraic_expression_value_l842_84233

theorem algebraic_expression_value (x : ℝ) (h : x = 2 * Real.sqrt 3 - 1) : x^2 + 2 * x - 3 = 8 :=
by 
  sorry

end algebraic_expression_value_l842_84233


namespace area_of_shaded_region_l842_84271

theorem area_of_shaded_region 
  (r R : ℝ)
  (hR : R = 9)
  (h : 2 * r = R) :
  π * R^2 - 3 * (π * r^2) = 20.25 * π :=
by
  sorry

end area_of_shaded_region_l842_84271


namespace find_m_l842_84227

theorem find_m 
  (x1 x2 : ℝ) 
  (m : ℝ)
  (h1 : x1 + x2 = m)
  (h2 : x1 * x2 = 2 * m - 1)
  (h3 : x1^2 + x2^2 = 7) : 
  m = 5 :=
by
  sorry

end find_m_l842_84227


namespace vertical_line_intersect_parabola_ex1_l842_84291

theorem vertical_line_intersect_parabola_ex1 (m : ℝ) (h : ∀ y : ℝ, (-4 * y^2 + 2*y + 3 = m) → false) :
  m = 13 / 4 :=
sorry

end vertical_line_intersect_parabola_ex1_l842_84291


namespace solve_inequality_l842_84275

theorem solve_inequality (a x : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 1 > 0) : 
  (-2 < a ∧ a < 1 → a < x ∧ x < 2 - a) ∧ 
  (a = 1 → False) ∧ 
  (1 < a ∧ a < 2 → 2 - a < x ∧ x < a) :=
by
  sorry

end solve_inequality_l842_84275


namespace mary_carrots_correct_l842_84265

def sandy_carrots := 8
def total_carrots := 14

def mary_carrots := total_carrots - sandy_carrots

theorem mary_carrots_correct : mary_carrots = 6 := by
  unfold mary_carrots
  unfold total_carrots
  unfold sandy_carrots
  sorry

end mary_carrots_correct_l842_84265


namespace sum_of_remainders_mod_53_l842_84209

theorem sum_of_remainders_mod_53 (x y z : ℕ) (hx : x % 53 = 36) (hy : y % 53 = 15) (hz : z % 53 = 7) : 
  (x + y + z) % 53 = 5 :=
by
  sorry

end sum_of_remainders_mod_53_l842_84209


namespace paperback_copies_sold_l842_84232

theorem paperback_copies_sold 
(H : ℕ)
(hardback_sold : H = 36000)
(P : ℕ)
(paperback_relation : P = 9 * H)
(total_copies : H + P = 440000) :
P = 324000 :=
sorry

end paperback_copies_sold_l842_84232


namespace pool_full_capacity_is_2000_l842_84244

-- Definitions based on the conditions given
def water_loss_per_jump : ℕ := 400 -- in ml
def jumps_before_cleaning : ℕ := 1000
def cleaning_threshold : ℚ := 0.80 -- 80%
def total_water_loss : ℕ := water_loss_per_jump * jumps_before_cleaning -- in ml
def water_loss_liters : ℚ := total_water_loss / 1000 -- converting ml to liters
def cleaning_loss_fraction : ℚ := 1 - cleaning_threshold -- 20% loss

-- The actual proof statement
theorem pool_full_capacity_is_2000 :
  (water_loss_liters : ℚ) / cleaning_loss_fraction = 2000 :=
by
  sorry

end pool_full_capacity_is_2000_l842_84244


namespace inequality_inequality_only_if_k_is_one_half_l842_84283

theorem inequality_inequality_only_if_k_is_one_half :
  (∀ t : ℝ, -1 < t ∧ t < 1 → (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1) ↔ k = 1 / 2 :=
by
  sorry

end inequality_inequality_only_if_k_is_one_half_l842_84283


namespace frosting_problem_l842_84270

-- Define the conditions
def cagney_rate := 1/15  -- Cagney's rate in cupcakes per second
def lacey_rate := 1/45   -- Lacey's rate in cupcakes per second
def total_time := 600  -- Total time in seconds (10 minutes)

-- Function to calculate the combined rate
def combined_rate (r1 r2 : ℝ) : ℝ := r1 + r2

-- Hypothesis combining the conditions
def hypothesis : Prop :=
  combined_rate cagney_rate lacey_rate = 1/11.25

-- Statement to prove: together they can frost 53 cupcakes within 10 minutes 
theorem frosting_problem : ∀ (total_time: ℝ) (hyp : hypothesis),
  total_time / (cagney_rate + lacey_rate) = 53 :=
by
  intro total_time hyp
  sorry

end frosting_problem_l842_84270


namespace proportion_solution_l842_84296

theorem proportion_solution (x : ℚ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by sorry

end proportion_solution_l842_84296


namespace solve_system_of_equations_l842_84230

theorem solve_system_of_equations (x y : ℝ) :
  (1 / 2 * x - 3 / 2 * y = -1) ∧ (2 * x + y = 3) → 
  (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_of_equations_l842_84230


namespace nested_sqrt_simplification_l842_84229

theorem nested_sqrt_simplification (y : ℝ) (hy : y ≥ 0) : 
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := 
sorry

end nested_sqrt_simplification_l842_84229


namespace height_difference_of_packings_l842_84210

theorem height_difference_of_packings :
  (let d := 12
   let n := 180
   let rowsA := n / 10
   let heightA := rowsA * d
   let height_of_hex_gap := (6 * Real.sqrt 3 : ℝ)
   let gaps := rowsA - 1
   let heightB := gaps * height_of_hex_gap + 2 * (d / 2)
   heightA - heightB) = 204 - 102 * Real.sqrt 3 :=
  sorry

end height_difference_of_packings_l842_84210


namespace partial_fraction_sum_zero_l842_84260

theorem partial_fraction_sum_zero (A B C D E F : ℚ) :
  (∀ x : ℚ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 → x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l842_84260


namespace total_books_l842_84200

variable (a : ℕ)

theorem total_books (h₁ : 5 = 5) (h₂ : a = a) : 5 + a = 5 + a :=
by
  sorry

end total_books_l842_84200


namespace cos_theta_minus_pi_six_l842_84243

theorem cos_theta_minus_pi_six (θ : ℝ) (h : Real.sin (θ + π / 3) = 2 / 3) : 
  Real.cos (θ - π / 6) = 2 / 3 :=
sorry

end cos_theta_minus_pi_six_l842_84243


namespace intersection_of_sets_l842_84292

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l842_84292


namespace integer_values_of_f_l842_84212

noncomputable def f (x : ℝ) : ℝ := (1 + x)^(1/3) + (3 - x)^(1/3)

theorem integer_values_of_f : 
  {x : ℝ | ∃ k : ℤ, f x = k} = {1 + Real.sqrt 5, 1 - Real.sqrt 5, 1 + (10/9) * Real.sqrt 3, 1 - (10/9) * Real.sqrt 3} :=
by
  sorry

end integer_values_of_f_l842_84212


namespace common_points_count_l842_84235

noncomputable def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
noncomputable def eq2 (x y : ℝ) : Prop := (x + 2 * y - 5) * (3 * x - 4 * y + 6) = 0

theorem common_points_count : 
  (∃ x1 y1 : ℝ, eq1 x1 y1 ∧ eq2 x1 y1) ∧
  (∃ x2 y2 : ℝ, eq1 x2 y2 ∧ eq2 x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)) ∧
  (∃ x3 y3 : ℝ, eq1 x3 y3 ∧ eq2 x3 y3 ∧ (x3 ≠ x1 ∧ x3 ≠ x2 ∧ y3 ≠ y1 ∧ y3 ≠ y2)) ∧ 
  (∃ x4 y4 : ℝ, eq1 x4 y4 ∧ eq2 x4 y4 ∧ (x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ y4 ≠ y1 ∧ y4 ≠ y2 ∧ y4 ≠ y3)) ∧ 
  ∀ x y : ℝ, (eq1 x y ∧ eq2 x y) → (((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))) :=
by
  sorry

end common_points_count_l842_84235


namespace coeff_comparison_l842_84221

def a_k (k : ℕ) : ℕ := (2 ^ k) * Nat.choose 100 k

theorem coeff_comparison :
  (Finset.filter (fun r => a_k r < a_k (r + 1)) (Finset.range 100)).card = 67 :=
by
  sorry

end coeff_comparison_l842_84221


namespace martys_journey_length_l842_84214

theorem martys_journey_length (x : ℝ) (h1 : x / 4 + 30 + x / 3 = x) : x = 72 :=
sorry

end martys_journey_length_l842_84214


namespace last_number_is_five_l842_84274

theorem last_number_is_five (seq : ℕ → ℕ) (h₀ : seq 0 = 5)
  (h₁ : ∀ n < 32, seq n + seq (n+1) + seq (n+2) + seq (n+3) + seq (n+4) + seq (n+5) = 29) :
  seq 36 = 5 :=
sorry

end last_number_is_five_l842_84274


namespace min_ratio_of_cylinder_cone_l842_84252

open Real

noncomputable def V1 (r : ℝ) : ℝ := 2 * π * r^3
noncomputable def V2 (R m r : ℝ) : ℝ := (1 / 3) * π * R^2 * m
noncomputable def geometric_constraint (R m r : ℝ) : Prop :=
  R / m = r / (sqrt ((m - r)^2 - r^2))

theorem min_ratio_of_cylinder_cone (r : ℝ) (hr : r > 0) : 
  ∃ R m, geometric_constraint R m r ∧ (V2 R m r) / (V1 r) = 4 / 3 := 
sorry

end min_ratio_of_cylinder_cone_l842_84252


namespace triangle_area_solution_l842_84248

noncomputable def triangle_area_problem 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  : ℝ := (1 / 2) * a * c * Real.sin B

theorem triangle_area_solution 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  (ha : a = 12)
  (hb : b = 6 * Real.sin (π / 3))
  (hA : A = π / 2)
  (hB : B = π / 3)
  (hC : C = π / 6) 
  : triangle_area_problem a b c A B C h1 h2 h3 = 18 * Real.sqrt 3 := by
  sorry

end triangle_area_solution_l842_84248


namespace find_percentage_l842_84242

theorem find_percentage (P : ℝ) (N : ℝ) (h1 : N = 140) (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end find_percentage_l842_84242


namespace circle_radius_through_focus_and_tangent_l842_84295

-- Define the given conditions of the problem
def ellipse_eq (x y : ℝ) : Prop := x^2 + 4 * y^2 = 16

-- State the problem as a theorem
theorem circle_radius_through_focus_and_tangent
  (x y : ℝ) (h : ellipse_eq x y) (r : ℝ) :
  r = 4 - 2 * Real.sqrt 3 :=
sorry

end circle_radius_through_focus_and_tangent_l842_84295


namespace ferry_speed_difference_l842_84237

open Nat

-- Define the time and speed of ferry P
def timeP := 3 -- hours
def speedP := 8 -- kilometers per hour

-- Define the distance of ferry P
def distanceP := speedP * timeP -- kilometers

-- Define the distance of ferry Q
def distanceQ := 3 * distanceP -- kilometers

-- Define the time of ferry Q
def timeQ := timeP + 5 -- hours

-- Define the speed of ferry Q
def speedQ := distanceQ / timeQ -- kilometers per hour

-- Define the speed difference
def speedDifference := speedQ - speedP -- kilometers per hour

-- The target theorem to prove
theorem ferry_speed_difference : speedDifference = 1 := by
  sorry

end ferry_speed_difference_l842_84237


namespace exponent_equality_l842_84282

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l842_84282


namespace fraction_remain_unchanged_l842_84225

theorem fraction_remain_unchanged (m n a b : ℚ) (h : n ≠ 0 ∧ b ≠ 0) : 
  (a / b = (a + m) / (b + n)) ↔ (a / b = m / n) :=
sorry

end fraction_remain_unchanged_l842_84225


namespace mean_properties_l842_84288

theorem mean_properties (a b c : ℝ) 
    (h1 : a + b + c = 36) 
    (h2 : a * b * c = 125) 
    (h3 : a * b + b * c + c * a = 93.75) : 
    a^2 + b^2 + c^2 = 1108.5 := 
by 
  sorry

end mean_properties_l842_84288


namespace double_root_divisors_l842_84223

theorem double_root_divisors (b3 b2 b1 s : ℤ) (h : 0 = (s^2) • (x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 50)) : 
  s = -5 ∨ s = -1 ∨ s = 1 ∨ s = 5 :=
by
  sorry

end double_root_divisors_l842_84223


namespace ivanov_family_net_worth_l842_84263

-- Define the financial values
def value_of_apartment := 3000000
def market_value_of_car := 900000
def bank_savings := 300000
def value_of_securities := 200000
def liquid_cash := 100000
def remaining_mortgage := 1500000
def car_loan := 500000
def debt_to_relatives := 200000

-- Calculate total assets and total liabilities
def total_assets := value_of_apartment + market_value_of_car + bank_savings + value_of_securities + liquid_cash
def total_liabilities := remaining_mortgage + car_loan + debt_to_relatives

-- Define the hypothesis and the final result of the net worth calculation
theorem ivanov_family_net_worth : total_assets - total_liabilities = 2300000 := by
  sorry

end ivanov_family_net_worth_l842_84263


namespace sum_of_six_terms_l842_84284

variable (a₁ a₂ a₃ a₄ a₅ a₆ q : ℝ)

-- Conditions
def geom_seq := a₂ = q * a₁ ∧ a₃ = q * a₂ ∧ a₄ = q * a₃ ∧ a₅ = q * a₄ ∧ a₆ = q * a₅
def cond₁ : Prop := a₁ + a₃ = 5 / 2
def cond₂ : Prop := a₂ + a₄ = 5 / 4

-- Problem Statement
theorem sum_of_six_terms : geom_seq a₁ a₂ a₃ a₄ a₅ a₆ q → cond₁ a₁ a₃ → cond₂ a₂ a₄ → 
  (a₁ * (1 - q^6) / (1 - q) = 63 / 16) := 
by 
  sorry

end sum_of_six_terms_l842_84284


namespace possible_values_of_reciprocal_l842_84293

theorem possible_values_of_reciprocal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ S, S = { x : ℝ | x >= 9 } ∧ (∃ x, x = (1/a + 1/b) ∧ x ∈ S) :=
sorry

end possible_values_of_reciprocal_l842_84293


namespace qingyang_2015_mock_exam_l842_84234

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def problem :=
  U = {1, 2, 3, 4, 5} ∧ A = {2, 3, 4} ∧ B = {2, 5} →
  B ∪ (U \ A) = {1, 2, 5}

theorem qingyang_2015_mock_exam (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) : problem U A B :=
by
  intros
  sorry

end qingyang_2015_mock_exam_l842_84234


namespace tree_height_at_end_of_4_years_l842_84226

theorem tree_height_at_end_of_4_years 
  (initial_growth : ℕ → ℕ)
  (height_7_years : initial_growth 7 = 64)
  (growth_pattern : ∀ n, initial_growth (n + 1) = 2 * initial_growth n) :
  initial_growth 4 = 8 :=
by
  sorry

end tree_height_at_end_of_4_years_l842_84226


namespace quadratic_root_a_value_l842_84297

theorem quadratic_root_a_value (a k : ℝ) (h1 : k = 65) (h2 : a * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : a = 2 :=
by
  sorry

end quadratic_root_a_value_l842_84297


namespace range_of_a_l842_84236

def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, 1 ≤ x ∧ x ≤ y → quadratic_function a x ≤ quadratic_function a y) : a ≤ 1 :=
sorry

end range_of_a_l842_84236


namespace slant_asymptote_and_sum_of_slope_and_intercept_l842_84255

noncomputable def f (x : ℚ) : ℚ := (3 * x^2 + 5 * x + 1) / (x + 2)

theorem slant_asymptote_and_sum_of_slope_and_intercept :
  (∀ x : ℚ, ∃ (m b : ℚ), (∃ r : ℚ, (r = f x ∧ (r + (m * x + b)) = f x)) ∧ m = 3 ∧ b = -1) →
  3 - 1 = 2 :=
by
  sorry

end slant_asymptote_and_sum_of_slope_and_intercept_l842_84255


namespace increase_by_percentage_l842_84208

def initial_value : ℕ := 550
def percentage_increase : ℚ := 0.35
def final_value : ℚ := 742.5

theorem increase_by_percentage :
  (initial_value : ℚ) * (1 + percentage_increase) = final_value := by
  sorry

end increase_by_percentage_l842_84208


namespace unique_digit_solution_l842_84220

-- Define the constraints as Lean predicates.
def sum_top_less_7 (A B C D E : ℕ) := A + B = (C + D + E) / 7
def sum_left_less_5 (A B C D E : ℕ) := A + C = (B + D + E) / 5

-- The main theorem statement asserting there is a unique solution.
theorem unique_digit_solution :
  ∃! (A B C D E : ℕ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 
  sum_top_less_7 A B C D E ∧ sum_left_less_5 A B C D E ∧
  (A, B, C, D, E) = (1, 2, 3, 4, 6) := sorry

end unique_digit_solution_l842_84220


namespace solve_xyz_l842_84277

theorem solve_xyz (a b c : ℝ) (h1 : a = y + z) (h2 : b = x + z) (h3 : c = x + y) 
                   (h4 : 0 < y) (h5 : 0 < z) (h6 : 0 < x)
                   (hab : b + c > a) (hbc : a + c > b) (hca : a + b > c) :
  x = (b - a + c)/2 ∧ y = (a - b + c)/2 ∧ z = (a + b - c)/2 :=
by
  sorry

end solve_xyz_l842_84277


namespace sqrt_x_minus_2_meaningful_l842_84267

theorem sqrt_x_minus_2_meaningful (x : ℝ) (h : 0 ≤ x - 2) : 2 ≤ x :=
by sorry

end sqrt_x_minus_2_meaningful_l842_84267


namespace solve_for_r_l842_84251

theorem solve_for_r (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ 
  r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 :=
by
  sorry

end solve_for_r_l842_84251


namespace dan_has_3_potatoes_left_l842_84294

-- Defining the number of potatoes Dan originally had
def original_potatoes : ℕ := 7

-- Defining the number of potatoes the rabbits ate
def potatoes_eaten : ℕ := 4

-- The theorem we want to prove: Dan has 3 potatoes left.
theorem dan_has_3_potatoes_left : original_potatoes - potatoes_eaten = 3 := by
  sorry

end dan_has_3_potatoes_left_l842_84294


namespace f_m_minus_1_pos_l842_84205

variable {R : Type*} [LinearOrderedField R]

def quadratic_function (x a : R) : R :=
  x^2 - x + a

theorem f_m_minus_1_pos {a m : R} (h_pos : 0 < a) (h_fm : quadratic_function m a < 0) :
  quadratic_function (m - 1 : R) a > 0 :=
sorry

end f_m_minus_1_pos_l842_84205


namespace prove_tirzah_handbags_l842_84211
noncomputable def tirzah_has_24_handbags (H : ℕ) : Prop :=
  let P := 26 -- number of purses
  let fakeP := P / 2 -- half of the purses are fake
  let authP := P - fakeP -- number of authentic purses
  let fakeH := H / 4 -- one quarter of the handbags are fake
  let authH := H - fakeH -- number of authentic handbags
  authP + authH = 31 -- total number of authentic items
  → H = 24 -- prove the number of handbags is 24

theorem prove_tirzah_handbags : ∃ H : ℕ, tirzah_has_24_handbags H :=
  by
    use 24
    -- Proof goes here
    sorry

end prove_tirzah_handbags_l842_84211


namespace set_intersection_l842_84272

open Set

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def intersection : Set ℕ := {1, 3}

theorem set_intersection : M ∩ N = intersection := by
  sorry

end set_intersection_l842_84272


namespace limit_of_f_at_infinity_l842_84253

open Filter
open Topology

variable (f : ℝ → ℝ)
variable (h_continuous : Continuous f)
variable (h_seq_limit : ∀ α > 0, Tendsto (fun n : ℕ => f (n * α)) atTop (nhds 0))

theorem limit_of_f_at_infinity : Tendsto f atTop (nhds 0) := by
  sorry

end limit_of_f_at_infinity_l842_84253


namespace inequality_solution_l842_84241

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (x^2 > x^(1 / 2)) ↔ (x > 1) :=
by
  sorry

end inequality_solution_l842_84241


namespace jackson_points_l842_84247

theorem jackson_points (team_total_points : ℕ)
                       (num_other_players : ℕ)
                       (average_points_other_players : ℕ)
                       (points_other_players: ℕ)
                       (points_jackson: ℕ)
                       (h_team_total_points : team_total_points = 65)
                       (h_num_other_players : num_other_players = 5)
                       (h_average_points_other_players : average_points_other_players = 6)
                       (h_points_other_players : points_other_players = num_other_players * average_points_other_players)
                       (h_points_total: points_jackson + points_other_players = team_total_points) :
  points_jackson = 35 :=
by
  -- proof will be done here
  sorry

end jackson_points_l842_84247


namespace cone_lateral_surface_area_l842_84222

theorem cone_lateral_surface_area (r : ℕ) (V : ℝ) (h l S : ℝ)
  (h_r : r = 6)
  (h_V : V = 30 * Real.pi)
  (h_volume : V = (1 / 3) * Real.pi * (r ^ 2) * h)
  (h_slant_height : l = Real.sqrt (r^2 + h^2))
  (h_lateral_surface_area : S = Real.pi * r * l) :
  S = 39 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l842_84222


namespace total_tour_time_l842_84290

-- Declare constants for distances
def distance1 : ℝ := 55
def distance2 : ℝ := 40
def distance3 : ℝ := 70
def extra_miles : ℝ := 10

-- Declare constants for speeds
def speed1_part1 : ℝ := 60
def speed1_part2 : ℝ := 40
def speed2 : ℝ := 45
def speed3_part1 : ℝ := 45
def speed3_part2 : ℝ := 35
def speed3_part3 : ℝ := 50
def return_speed : ℝ := 55

-- Declare constants for stop times
def stop1 : ℝ := 1
def stop2 : ℝ := 1.5
def stop3 : ℝ := 2

-- Prove the total time required for the tour
theorem total_tour_time :
  (30 / speed1_part1) + (25 / speed1_part2) + stop1 +
  (distance2 / speed2) + stop2 +
  (20 / speed3_part1) + (30 / speed3_part2) + (20 / speed3_part3) + stop3 +
  ((distance1 + distance2 + distance3 + extra_miles) / return_speed) = 11.40 :=
by
  sorry

end total_tour_time_l842_84290


namespace vector_identity_l842_84266

def vec_a : ℝ × ℝ := (2, 2)
def vec_b : ℝ × ℝ := (-1, 3)

theorem vector_identity : 2 • vec_a - vec_b = (5, 1) := by
  sorry

end vector_identity_l842_84266


namespace pradeep_max_marks_l842_84287

theorem pradeep_max_marks (M : ℝ) 
  (pass_condition : 0.35 * M = 210) : M = 600 :=
sorry

end pradeep_max_marks_l842_84287


namespace unique_k_exists_l842_84281

theorem unique_k_exists (k : ℕ) (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (a^2 + b^2 = k * a * b) ↔ k = 2 := sorry

end unique_k_exists_l842_84281


namespace find_a_l842_84249

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.sin x

theorem find_a (a : ℝ) : (∀ f', f' = (fun x => a * Real.exp x - Real.cos x) → f' 0 = 0) → a = 1 :=
by
  intros h
  specialize h (fun x => a * Real.exp x - Real.cos x) rfl
  sorry  -- proof is omitted

end find_a_l842_84249


namespace common_ratio_value_l842_84268

theorem common_ratio_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) → (x + y + z = 0 ∨ x + y + z ≠ 0) → ((x + y) / z = -1 ∨ (x + y) / z = 2) :=
by
  sorry

end common_ratio_value_l842_84268


namespace triangle_acute_l842_84245

theorem triangle_acute (A B C : ℝ) (h1 : A = 2 * (180 / 9)) (h2 : B = 3 * (180 / 9)) (h3 : C = 4 * (180 / 9)) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_acute_l842_84245


namespace total_earrings_l842_84201

-- Definitions based on the given conditions
def bella_earrings : ℕ := 10
def monica_earrings : ℕ := 4 * bella_earrings
def rachel_earrings : ℕ := monica_earrings / 2
def olivia_earrings : ℕ := bella_earrings + monica_earrings + rachel_earrings + 5

-- The theorem to prove the total number of earrings
theorem total_earrings : bella_earrings + monica_earrings + rachel_earrings + olivia_earrings = 145 := by
  sorry

end total_earrings_l842_84201


namespace find_k_for_parallel_vectors_l842_84262

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem find_k_for_parallel_vectors 
  (h_a : a = (2, -1)) 
  (h_b : b = (1, 1)) 
  (h_c : c = (-5, 1)) 
  (h_parallel : vector_parallel (a.1 + k * b.1, a.2 + k * b.2) c) : 
  k = 1 / 2 :=
by
  unfold vector_parallel at h_parallel
  simp at h_parallel
  sorry

end find_k_for_parallel_vectors_l842_84262


namespace abc_equality_l842_84298

theorem abc_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                      (h : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equality_l842_84298


namespace twenty_percent_correct_l842_84299

def certain_number := 400
def forty_percent (x : ℕ) : ℕ := 40 * x / 100
def twenty_percent_of_certain_number (x : ℕ) : ℕ := 20 * x / 100

theorem twenty_percent_correct : 
  (∃ x : ℕ, forty_percent x = 160) → twenty_percent_of_certain_number certain_number = 80 :=
by
  sorry

end twenty_percent_correct_l842_84299
