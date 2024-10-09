import Mathlib

namespace convex_polygon_sides_l912_91254

theorem convex_polygon_sides (S : ℝ) (n : ℕ) (a₁ a₂ a₃ a₄ : ℝ) 
    (h₁ : S = 4320) 
    (h₂ : a₁ = 120) 
    (h₃ : a₂ = 120) 
    (h₄ : a₃ = 120) 
    (h₅ : a₄ = 120) 
    (h_sum : S = 180 * (n - 2)) :
    n = 26 :=
by
  sorry

end convex_polygon_sides_l912_91254


namespace revenue_times_l912_91238

noncomputable def revenue_ratio (D : ℝ) : ℝ :=
  let revenue_Nov := (2 / 5) * D
  let revenue_Jan := (1 / 3) * revenue_Nov
  let average := (revenue_Nov + revenue_Jan) / 2
  D / average

theorem revenue_times (D : ℝ) (hD : D ≠ 0) : revenue_ratio D = 3.75 :=
by
  -- skipped proof
  sorry

end revenue_times_l912_91238


namespace find_a1_l912_91219

-- Define the sequence
def seq (a : ℕ → ℝ) := ∀ n : ℕ, 0 < n → a n = (1/2) * a (n + 1)

-- Given conditions
def a3_value (a : ℕ → ℝ) := a 3 = 12

-- Theorem statement
theorem find_a1 (a : ℕ → ℝ) (h_seq : seq a) (h_a3 : a3_value a) : a 1 = 3 :=
by
  sorry

end find_a1_l912_91219


namespace total_animals_is_63_l912_91244

def zoo_animals (penguins polar_bears total : ℕ) : Prop :=
  (penguins = 21) ∧
  (polar_bears = 2 * penguins) ∧
  (total = penguins + polar_bears)

theorem total_animals_is_63 :
  ∃ (penguins polar_bears total : ℕ), zoo_animals penguins polar_bears total ∧ total = 63 :=
by {
  sorry
}

end total_animals_is_63_l912_91244


namespace value_of_a7_l912_91277

-- Define the geometric sequence and its properties
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the conditions of the problem
variables (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n : ℕ, a n > 0) (h_product : a 3 * a 11 = 16)

-- Conjecture that we aim to prove
theorem value_of_a7 : a 7 = 4 :=
by {
  sorry
}

end value_of_a7_l912_91277


namespace problem_1_problem_2_problem_3_l912_91287

def M := {n : ℕ | 0 < n ∧ n < 1000}

def circ (a b : ℕ) : ℕ :=
  if a * b < 1000 then a * b
  else 
    let k := (a * b) / 1000
    let r := (a * b) % 1000
    if k + r < 1000 then k + r
    else (k + r) % 1000 + 1

theorem problem_1 : circ 559 758 = 146 := 
by
  sorry

theorem problem_2 : ∃ (x : ℕ) (h : x ∈ M), circ 559 x = 1 ∧ x = 361 :=
by
  sorry

theorem problem_3 : ∀ (a b c : ℕ) (h₁ : a ∈ M) (h₂ : b ∈ M) (h₃ : c ∈ M), circ a (circ b c) = circ (circ a b) c :=
by
  sorry

end problem_1_problem_2_problem_3_l912_91287


namespace find_a_minus_b_l912_91265

theorem find_a_minus_b (a b x y : ℤ)
  (h_x : x = 1)
  (h_y : y = 1)
  (h1 : a * x + b * y = 2)
  (h2 : x - b * y = 3) :
  a - b = 6 := by
  subst h_x
  subst h_y
  simp at h1 h2
  have h_b: b = -2 := by linarith
  have h_a: a = 4 := by linarith
  rw [h_a, h_b]
  norm_num

end find_a_minus_b_l912_91265


namespace tyre_punctures_deflation_time_l912_91288

theorem tyre_punctures_deflation_time :
  (1 / (1 / 9 + 1 / 6)) = 3.6 :=
by
  sorry

end tyre_punctures_deflation_time_l912_91288


namespace largest_subset_size_l912_91296

theorem largest_subset_size (T : Finset ℕ) (h : ∀ x ∈ T, ∀ y ∈ T, x ≠ y → (x - y) % 2021 ≠ 5 ∧ (x - y) % 2021 ≠ 8) :
  T.card ≤ 918 := sorry

end largest_subset_size_l912_91296


namespace numberOfWaysToChooseLeadershipStructure_correct_l912_91213

noncomputable def numberOfWaysToChooseLeadershipStructure : ℕ :=
  12 * 11 * 10 * Nat.choose 9 3 * Nat.choose 6 3

theorem numberOfWaysToChooseLeadershipStructure_correct :
  numberOfWaysToChooseLeadershipStructure = 221760 :=
by
  simp [numberOfWaysToChooseLeadershipStructure]
  -- Add detailed simplification/proof steps here if required
  sorry

end numberOfWaysToChooseLeadershipStructure_correct_l912_91213


namespace total_number_of_cookies_l912_91212

open Nat -- Open the natural numbers namespace to work with natural number operations

def n_bags : Nat := 7
def cookies_per_bag : Nat := 2
def total_cookies : Nat := n_bags * cookies_per_bag

theorem total_number_of_cookies : total_cookies = 14 := by
  sorry

end total_number_of_cookies_l912_91212


namespace number_of_routes_from_P_to_Q_is_3_l912_91264

-- Definitions of the nodes and paths
inductive Node
| P | Q | R | S | T | U | V
deriving DecidableEq, Repr

-- Definition of paths between nodes based on given conditions
def leads_to : Node → Node → Prop
| Node.P, Node.R => True
| Node.P, Node.S => True
| Node.R, Node.T => True
| Node.R, Node.U => True
| Node.S, Node.Q => True
| Node.T, Node.Q => True
| Node.U, Node.V => True
| Node.V, Node.Q => True
| _, _ => False

-- Proof statement: the number of different routes from P to Q
theorem number_of_routes_from_P_to_Q_is_3 : 
  ∃ (n : ℕ), n = 3 ∧ (∀ (route_count : ℕ), route_count = n → 
  ((leads_to Node.P Node.R ∧ leads_to Node.R Node.T ∧ leads_to Node.T Node.Q) ∨ 
   (leads_to Node.P Node.R ∧ leads_to Node.R Node.U ∧ leads_to Node.U Node.V ∧ leads_to Node.V Node.Q) ∨
   (leads_to Node.P Node.S ∧ leads_to Node.S Node.Q))) :=
by
  -- Placeholder proof
  sorry

end number_of_routes_from_P_to_Q_is_3_l912_91264


namespace profit_percent_l912_91261

-- Definitions based on the conditions in the problem
def marked_price_per_pen := ℝ
def total_pens := 52
def cost_equivalent_pens := 46
def discount_percentage := 1 / 100

-- Values calculated from conditions
def cost_price (P : ℝ) := cost_equivalent_pens * P
def selling_price_per_pen (P : ℝ) := P * (1 - discount_percentage)
def total_selling_price (P : ℝ) := total_pens * selling_price_per_pen P

-- The proof statement
theorem profit_percent (P : ℝ) (hP : P > 0) :
  ((total_selling_price P - cost_price P) / (cost_price P)) * 100 = 11.91 := by
    sorry

end profit_percent_l912_91261


namespace bagels_count_l912_91234

def total_items : ℕ := 90
def bread_rolls : ℕ := 49
def croissants : ℕ := 19

def bagels : ℕ := total_items - (bread_rolls + croissants)

theorem bagels_count : bagels = 22 :=
by
  sorry

end bagels_count_l912_91234


namespace no_integer_triplets_satisfying_eq_l912_91222

theorem no_integer_triplets_satisfying_eq (x y z : ℤ) : 3 * x^2 + 7 * y^2 ≠ z^4 := 
by {
  sorry
}

end no_integer_triplets_satisfying_eq_l912_91222


namespace problem_l912_91223

-- Step 1: Define the transformation functions
def rotate_90_counterclockwise (h k x y : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

-- Step 2: Define the given problem condition
theorem problem (a b : ℝ) :
  rotate_90_counterclockwise 2 3 (reflect_y_eq_x 5 1).fst (reflect_y_eq_x 5 1).snd = (a, b) →
  b - a = 0 :=
by
  intro h
  sorry

end problem_l912_91223


namespace minimum_a_plus_2b_no_a_b_such_that_l912_91209

noncomputable def minimum_value (a b : ℝ) :=
  a + 2 * b

theorem minimum_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  minimum_value a b ≥ 6 :=
sorry

theorem no_a_b_such_that (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  a^2 + 4 * b^2 ≠ 17 :=
sorry

end minimum_a_plus_2b_no_a_b_such_that_l912_91209


namespace magazines_in_third_pile_l912_91270

-- Define the number of magazines in each pile.
def pile1 := 3
def pile2 := 4
def pile4 := 9
def pile5 := 13

-- Define the differences between the piles.
def diff2_1 := pile2 - pile1  -- Difference between second and first pile
def diff4_2 := pile4 - pile2  -- Difference between fourth and second pile

-- Assume the pattern continues with differences increasing by 4.
def diff3_2 := diff2_1 + 4    -- Difference between third and second pile

-- Define the number of magazines in the third pile.
def pile3 := pile2 + diff3_2

-- Theorem stating the number of magazines in the third pile.
theorem magazines_in_third_pile : pile3 = 9 := by sorry

end magazines_in_third_pile_l912_91270


namespace min_value_a_sq_plus_b_sq_l912_91269

theorem min_value_a_sq_plus_b_sq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x > 0 → y > 0 → (x - 1)^3 + (y - 1)^3 ≥ 3 * (2 - x - y) → x^2 + y^2 ≥ m) :=
by
  sorry

end min_value_a_sq_plus_b_sq_l912_91269


namespace shorter_side_of_rectangle_l912_91291

theorem shorter_side_of_rectangle (a b : ℕ) (h_perimeter : 2 * a + 2 * b = 62) (h_area : a * b = 240) : b = 15 :=
by
  sorry

end shorter_side_of_rectangle_l912_91291


namespace interest_rate_is_4_percent_l912_91216

variable (P A n : ℝ)
variable (r : ℝ)
variable (n_pos : n ≠ 0)

-- Define the conditions
def principal : ℝ := P
def amount_after_n_years : ℝ := A
def years : ℝ := n
def interest_rate : ℝ := r

-- The compound interest formula
def compound_interest (P A r : ℝ) (n : ℝ) : Prop :=
  A = P * (1 + r) ^ n

-- The Lean theorem statement
theorem interest_rate_is_4_percent
  (P_val : principal = 7500)
  (A_val : amount_after_n_years = 8112)
  (n_val : years = 2)
  (h : compound_interest P A r n) :
  r = 0.04 :=
sorry

end interest_rate_is_4_percent_l912_91216


namespace larry_substitution_l912_91206

theorem larry_substitution (a b c d e : ℤ)
  (h_a : a = 2)
  (h_b : b = 5)
  (h_c : c = 3)
  (h_d : d = 4)
  (h_expr1 : a + b - c - d * e = 4 - 4 * e)
  (h_expr2 : a + (b - (c - (d * e))) = 4 + 4 * e) :
  e = 0 :=
by
  sorry

end larry_substitution_l912_91206


namespace cost_of_ox_and_sheep_l912_91294

variable (x y : ℚ)

theorem cost_of_ox_and_sheep :
  (5 * x + 2 * y = 10) ∧ (2 * x + 8 * y = 8) → (x = 16 / 9 ∧ y = 5 / 9) :=
by
  sorry

end cost_of_ox_and_sheep_l912_91294


namespace y_mul_k_is_perfect_square_l912_91282

-- Defining y as given in the problem with its prime factorization
def y : Nat := 3^4 * (2^2)^5 * 5^6 * (2 * 3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Since the question asks for an integer k (in this case 75) such that y * k is a perfect square
def k : Nat := 75

-- The statement that needs to be proved
theorem y_mul_k_is_perfect_square : ∃ n : Nat, (y * k) = n^2 := 
by
  sorry

end y_mul_k_is_perfect_square_l912_91282


namespace quotient_of_division_l912_91248

theorem quotient_of_division (dividend divisor remainder quotient : ℕ)
  (h_dividend : dividend = 15)
  (h_divisor : divisor = 3)
  (h_remainder : remainder = 3)
  (h_relation : dividend = divisor * quotient + remainder) :
  quotient = 4 :=
by sorry

end quotient_of_division_l912_91248


namespace benny_added_march_l912_91281

theorem benny_added_march :
  let january := 19 
  let february := 19
  let march_total := 46
  (march_total - (january + february) = 8) :=
by
  let january := 19
  let february := 19
  let march_total := 46
  sorry

end benny_added_march_l912_91281


namespace find_bicycle_speed_l912_91235

def distanceAB := 40 -- Distance from A to B in km
def speed_walk := 6 -- Speed of the walking tourist in km/h
def distance_ahead := 5 -- Distance by which the second tourist is ahead initially in km
def speed_car := 24 -- Speed of the car in km/h
def meeting_time := 2 -- Time after departure when they meet in hours

theorem find_bicycle_speed (v : ℝ) : 
  (distanceAB = 40 ∧ speed_walk = 6 ∧ distance_ahead = 5 ∧ speed_car = 24 ∧ meeting_time = 2) →
  (v = 9) :=
by 
sorry

end find_bicycle_speed_l912_91235


namespace only_setB_is_proportional_l912_91251

-- Definitions for the line segments
def setA := (3, 4, 5, 6)
def setB := (5, 15, 2, 6)
def setC := (4, 8, 3, 5)
def setD := (8, 4, 1, 3)

-- Definition to check if a set of line segments is proportional
def is_proportional (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c, d) := s
  a * d = b * c

-- Theorem proving that the only proportional set is set B
theorem only_setB_is_proportional :
  is_proportional setA = false ∧
  is_proportional setB = true ∧
  is_proportional setC = false ∧
  is_proportional setD = false :=
by
  sorry

end only_setB_is_proportional_l912_91251


namespace solve_money_conditions_l912_91205

theorem solve_money_conditions 
  (a b : ℝ)
  (h1 : b - 4 * a < 78)
  (h2 : 6 * a - b = 36) :
  a < 57 ∧ b > -36 :=
sorry

end solve_money_conditions_l912_91205


namespace largest_polygon_area_l912_91274

variable (area : ℕ → ℝ)

def polygon_A_area : ℝ := 6
def polygon_B_area : ℝ := 3 + 4 * 0.5
def polygon_C_area : ℝ := 4 + 5 * 0.5
def polygon_D_area : ℝ := 7
def polygon_E_area : ℝ := 2 + 6 * 0.5

theorem largest_polygon_area : polygon_D_area = max (max (max polygon_A_area polygon_B_area) polygon_C_area) polygon_E_area :=
by
  sorry

end largest_polygon_area_l912_91274


namespace ribbon_problem_l912_91285

variable (Ribbon1 Ribbon2 : ℕ)
variable (L : ℕ)

theorem ribbon_problem
    (h1 : Ribbon1 = 8)
    (h2 : ∀ L, L > 0 → Ribbon1 % L = 0 → Ribbon2 % L = 0)
    (h3 : ∀ k, (k > 0 ∧ Ribbon1 % k = 0 ∧ Ribbon2 % k = 0) → k ≤ 8) :
    Ribbon2 = 8 := by
  sorry

end ribbon_problem_l912_91285


namespace book_arrangement_l912_91241

theorem book_arrangement : (Nat.choose 7 3 = 35) :=
by
  sorry

end book_arrangement_l912_91241


namespace hazel_sold_18_cups_to_kids_l912_91272

theorem hazel_sold_18_cups_to_kids:
  ∀ (total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup: ℕ),
     total_cups = 56 →
     cups_sold_construction = 28 →
     crew_remaining = total_cups - cups_sold_construction →
     last_cup = 1 →
     crew_remaining = cups_sold_kids + (cups_sold_kids / 2) + last_cup →
     cups_sold_kids = 18 :=
by
  intros total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup h_total h_construction h_remaining h_last h_equation
  sorry

end hazel_sold_18_cups_to_kids_l912_91272


namespace haley_cider_pints_l912_91259

noncomputable def apples_per_farmhand_per_hour := 240
noncomputable def working_hours := 5
noncomputable def total_farmhands := 6

noncomputable def golden_delicious_per_pint := 20
noncomputable def pink_lady_per_pint := 40
noncomputable def golden_delicious_ratio := 1
noncomputable def pink_lady_ratio := 2

noncomputable def total_apples := total_farmhands * apples_per_farmhand_per_hour * working_hours
noncomputable def total_parts := golden_delicious_ratio + pink_lady_ratio

noncomputable def golden_delicious_apples := total_apples / total_parts
noncomputable def pink_lady_apples := golden_delicious_apples * pink_lady_ratio

noncomputable def pints_golden_delicious := golden_delicious_apples / golden_delicious_per_pint
noncomputable def pints_pink_lady := pink_lady_apples / pink_lady_per_pint

theorem haley_cider_pints : 
  total_apples = 7200 → 
  golden_delicious_apples = 2400 → 
  pink_lady_apples = 4800 → 
  pints_golden_delicious = 120 → 
  pints_pink_lady = 120 → 
  pints_golden_delicious = pints_pink_lady →
  pints_golden_delicious = 120 :=
by
  sorry

end haley_cider_pints_l912_91259


namespace solve_trig_equation_l912_91246

open Real

theorem solve_trig_equation (k : ℕ) :
    (∀ x, 8.459 * cos x^2 * cos (x^2) * (tan (x^2) + 2 * tan x) + tan x^3 * (1 - sin (x^2)^2) * (2 - tan x * tan (x^2)) = 0) ↔
    (∃ k : ℕ, x = -1 + sqrt (π * k + 1) ∨ x = -1 - sqrt (π * k + 1)) :=
sorry

end solve_trig_equation_l912_91246


namespace digit_ends_with_l912_91257

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end digit_ends_with_l912_91257


namespace problem_statement_l912_91262

def Delta (a b : ℝ) : ℝ := a^2 - b

theorem problem_statement : Delta (2 ^ (Delta 5 8)) (4 ^ (Delta 2 7)) = 17179869183.984375 := by
  sorry

end problem_statement_l912_91262


namespace coins_count_l912_91290

variable (x : ℕ)

def total_value : ℕ → ℕ := λ x => x + (x * 50) / 100 + (x * 25) / 100

theorem coins_count (h : total_value x = 140) : x = 80 :=
sorry

end coins_count_l912_91290


namespace carpet_area_l912_91242

-- Definitions
def Rectangle1 (length1 width1 : ℕ) : Prop :=
  length1 = 12 ∧ width1 = 9

def Rectangle2 (length2 width2 : ℕ) : Prop :=
  length2 = 6 ∧ width2 = 9

def feet_to_yards (feet : ℕ) : ℕ :=
  feet / 3

-- Statement to prove
theorem carpet_area (length1 width1 length2 width2 : ℕ) (h1 : Rectangle1 length1 width1) (h2 : Rectangle2 length2 width2) :
  feet_to_yards (length1 * width1) / 3 + feet_to_yards (length2 * width2) / 3 = 18 :=
by
  sorry

end carpet_area_l912_91242


namespace no_nat_num_divisible_l912_91220

open Nat

theorem no_nat_num_divisible : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 := sorry

end no_nat_num_divisible_l912_91220


namespace tangent_fraction_15_degrees_l912_91226

theorem tangent_fraction_15_degrees : (1 + Real.tan (Real.pi / 12 )) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end tangent_fraction_15_degrees_l912_91226


namespace charlie_first_week_usage_l912_91210

noncomputable def data_used_week1 : ℕ :=
  let data_plan := 8
  let week2_usage := 3
  let week3_usage := 5
  let week4_usage := 10
  let total_extra_cost := 120
  let cost_per_gb_extra := 10
  let total_data_used := data_plan + (total_extra_cost / cost_per_gb_extra)
  let total_data_week_2_3_4 := week2_usage + week3_usage + week4_usage
  total_data_used - total_data_week_2_3_4

theorem charlie_first_week_usage : data_used_week1 = 2 :=
by
  sorry

end charlie_first_week_usage_l912_91210


namespace repeating_decimal_to_fraction_l912_91228

noncomputable def repeating_decimal_sum (x y z : ℚ) : ℚ := x + y + z

theorem repeating_decimal_to_fraction :
  let x := 4 / 33
  let y := 34 / 999
  let z := 567 / 99999
  repeating_decimal_sum x y z = 134255 / 32929667 := by
  -- proofs are omitted
  sorry

end repeating_decimal_to_fraction_l912_91228


namespace marbles_in_jar_is_144_l912_91286

noncomputable def marbleCount (M : ℕ) : Prop :=
  M / 16 - M / 18 = 1

theorem marbles_in_jar_is_144 : ∃ M : ℕ, marbleCount M ∧ M = 144 :=
by
  use 144
  unfold marbleCount
  sorry

end marbles_in_jar_is_144_l912_91286


namespace sandy_nickels_remaining_l912_91280

def original_nickels : ℕ := 31
def nickels_borrowed : ℕ := 20

theorem sandy_nickels_remaining : (original_nickels - nickels_borrowed) = 11 :=
by
  sorry

end sandy_nickels_remaining_l912_91280


namespace max_side_of_triangle_l912_91258

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end max_side_of_triangle_l912_91258


namespace all_numbers_positive_l912_91256

theorem all_numbers_positive (n : ℕ) (a : Fin (2 * n + 1) → ℝ) 
  (h : ∀ S : Finset (Fin (2 * n + 1)), 
        S.card = n + 1 → 
        S.sum a > (Finset.univ \ S).sum a) : 
  ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l912_91256


namespace value_of_T_l912_91230

theorem value_of_T (T : ℝ) (h : (1 / 3) * (1 / 6) * T = (1 / 4) * (1 / 8) * 120) : T = 67.5 :=
sorry

end value_of_T_l912_91230


namespace x_intercept_of_line_l912_91271

theorem x_intercept_of_line : ∀ x y : ℝ, 2 * x + 3 * y = 6 → y = 0 → x = 3 :=
by
  intros x y h_line h_y_zero
  sorry

end x_intercept_of_line_l912_91271


namespace eleven_hash_five_l912_91283

def my_op (r s : ℝ) : ℝ := sorry

axiom op_cond1 : ∀ r : ℝ, my_op r 0 = r
axiom op_cond2 : ∀ r s : ℝ, my_op r s = my_op s r
axiom op_cond3 : ∀ r s : ℝ, my_op (r + 1) s = (my_op r s) + s + 1

theorem eleven_hash_five : my_op 11 5 = 71 :=
by {
    sorry
}

end eleven_hash_five_l912_91283


namespace inequality_transformation_l912_91263

theorem inequality_transformation (x y a : ℝ) (hxy : x < y) (ha : a < 1) : x + a < y + 1 := by
  sorry

end inequality_transformation_l912_91263


namespace tom_has_18_apples_l912_91229

-- Definitions based on conditions
def phillip_apples : ℕ := 40
def ben_apples : ℕ := phillip_apples + 8
def tom_apples : ℕ := (3 * ben_apples) / 8

-- Theorem stating Tom has 18 apples given the conditions
theorem tom_has_18_apples : tom_apples = 18 :=
sorry

end tom_has_18_apples_l912_91229


namespace sequence_bounds_l912_91266

theorem sequence_bounds :
    ∀ (a : ℕ → ℝ), a 0 = 5 → (∀ n : ℕ, a (n + 1) = a n + 1 / a n) → 45 < a 1000 ∧ a 1000 < 45.1 :=
by
  intros a h0 h_rec
  sorry

end sequence_bounds_l912_91266


namespace complex_number_second_quadrant_l912_91200

theorem complex_number_second_quadrant 
  : (2 + 3 * Complex.I) / (1 - Complex.I) ∈ { z : Complex | z.re < 0 ∧ z.im > 0 } := 
by
  sorry

end complex_number_second_quadrant_l912_91200


namespace energy_stick_difference_l912_91249

variable (B D : ℕ)

theorem energy_stick_difference (h1 : B = D + 17) : 
  let B' := B - 3
  let D' := D + 3
  D' < B' →
  (B' - D') = 11 :=
by
  sorry

end energy_stick_difference_l912_91249


namespace tank_filling_time_l912_91275

theorem tank_filling_time
  (T : ℕ) (Rₐ R_b R_c : ℕ) (C : ℕ)
  (hRₐ : Rₐ = 40) (hR_b : R_b = 30) (hR_c : R_c = 20) (hC : C = 950)
  (h_cycle : T = 1 + 1 + 1) : 
  T * (C / (Rₐ + R_b - R_c)) - 1 = 56 :=
by
  sorry

end tank_filling_time_l912_91275


namespace gcd_three_numbers_l912_91299

def a : ℕ := 13680
def b : ℕ := 20400
def c : ℕ := 47600

theorem gcd_three_numbers (a b c : ℕ) : Nat.gcd (Nat.gcd a b) c = 80 :=
by
  sorry

end gcd_three_numbers_l912_91299


namespace right_triangle_of_ratio_and_right_angle_l912_91253

-- Define the sides and the right angle condition based on the problem conditions
variable (x : ℝ) (hx : 0 < x)

-- Variables for the sides in the given ratio
def a := 3 * x
def b := 4 * x
def c := 5 * x

-- The proposition we need to prove
theorem right_triangle_of_ratio_and_right_angle (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by sorry  -- Proof not required as per instructions

end right_triangle_of_ratio_and_right_angle_l912_91253


namespace values_of_y_satisfy_quadratic_l912_91278

theorem values_of_y_satisfy_quadratic :
  (∃ (x y : ℝ), 3 * x^2 + 4 * x + 7 * y + 2 = 0 ∧ 3 * x + 2 * y + 4 = 0) →
  (∃ (y : ℝ), 4 * y^2 + 29 * y + 6 = 0) :=
by sorry

end values_of_y_satisfy_quadratic_l912_91278


namespace quadratic_real_roots_iff_l912_91203

theorem quadratic_real_roots_iff (α : ℝ) : (∃ x : ℝ, x^2 - 2 * x + α = 0) ↔ α ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_l912_91203


namespace total_daisies_l912_91240

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end total_daisies_l912_91240


namespace find_cookies_per_tray_l912_91284

def trays_baked_per_day := 2
def days_of_baking := 6
def cookies_eaten_by_frank := 1
def cookies_eaten_by_ted := 4
def cookies_left := 134

theorem find_cookies_per_tray (x : ℕ) (h : 12 * x - 10 = 134) : x = 12 :=
by
  sorry

end find_cookies_per_tray_l912_91284


namespace largest_multiple_of_8_less_than_100_l912_91245

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), n < 100 ∧ 8 ∣ n ∧ ∀ (m : ℕ), m < 100 ∧ 8 ∣ m → m ≤ n :=
sorry

end largest_multiple_of_8_less_than_100_l912_91245


namespace express_as_scientific_notation_l912_91224

-- Definitions
def billion : ℝ := 10^9
def amount : ℝ := 850 * billion

-- Statement
theorem express_as_scientific_notation : amount = 8.5 * 10^11 :=
by
  sorry

end express_as_scientific_notation_l912_91224


namespace John_pushup_count_l912_91239

-- Definitions arising from conditions
def Zachary_pushups : ℕ := 51
def David_pushups : ℕ := Zachary_pushups + 22
def John_pushups : ℕ := David_pushups - 4

-- Theorem statement
theorem John_pushup_count : John_pushups = 69 := 
by 
  sorry

end John_pushup_count_l912_91239


namespace jill_spent_on_other_items_l912_91202

theorem jill_spent_on_other_items {T : ℝ} (h₁ : T > 0)
    (h₁ : 0.5 * T + 0.2 * T + O * T / 100 = T)
    (h₂ : 0.04 * 0.5 * T = 0.02 * T)
    (h₃ : 0 * 0.2 * T = 0)
    (h₄ : 0.08 * O * T / 100 = 0.0008 * O * T)
    (h₅ : 0.044 * T = 0.02 * T + 0 + 0.0008 * O * T) :
  O = 30 := 
sorry

end jill_spent_on_other_items_l912_91202


namespace fly_dist_ceiling_eq_sqrt255_l912_91218

noncomputable def fly_distance_from_ceiling : ℝ :=
  let x := 3
  let y := 5
  let d := 17
  let z := Real.sqrt (d^2 - (x^2 + y^2))
  z

theorem fly_dist_ceiling_eq_sqrt255 :
  fly_distance_from_ceiling = Real.sqrt 255 :=
by
  sorry

end fly_dist_ceiling_eq_sqrt255_l912_91218


namespace bonus_points_amount_l912_91297

def points_per_10_dollars : ℕ := 50

def beef_price : ℕ := 11
def beef_quantity : ℕ := 3

def fruits_vegetables_price : ℕ := 4
def fruits_vegetables_quantity : ℕ := 8

def spices_price : ℕ := 6
def spices_quantity : ℕ := 3

def other_groceries_total : ℕ := 37

def total_points : ℕ := 850

def total_spent : ℕ :=
  (beef_price * beef_quantity) +
  (fruits_vegetables_price * fruits_vegetables_quantity) +
  (spices_price * spices_quantity) +
  other_groceries_total

def points_from_spending : ℕ :=
  (total_spent / 10) * points_per_10_dollars

theorem bonus_points_amount :
  total_spent > 100 → total_points - points_from_spending = 250 :=
by
  sorry

end bonus_points_amount_l912_91297


namespace derivative_of_even_function_is_odd_l912_91231

variables {R : Type*}

-- Definitions and Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem derivative_of_even_function_is_odd (f g : ℝ → ℝ) (h1 : even_function f) (h2 : ∀ x, deriv f x = g x) : odd_function g :=
sorry

end derivative_of_even_function_is_odd_l912_91231


namespace angie_age_problem_l912_91260

theorem angie_age_problem (a certain_number : ℕ) 
  (h1 : 2 * 8 + certain_number = 20) : 
  certain_number = 4 :=
by 
  sorry

end angie_age_problem_l912_91260


namespace which_is_linear_l912_91295

-- Define what it means to be a linear equation in two variables
def is_linear_equation_in_two_vars (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y = (a * x + b * y = c)

-- Define each of the given equations
def equation_A (x y : ℝ) : Prop := x / 2 + 3 * y = 2
def equation_B (x y : ℝ) : Prop := x / 2 + 1 = 3 * x * y
def equation_C (x y : ℝ) : Prop := 2 * x + 1 = 3 * x
def equation_D (x y : ℝ) : Prop := 3 * x + 2 * y^2 = 1

-- Theorem stating which equation is linear in two variables
theorem which_is_linear : 
  is_linear_equation_in_two_vars equation_A ∧ 
  ¬ is_linear_equation_in_two_vars equation_B ∧ 
  ¬ is_linear_equation_in_two_vars equation_C ∧ 
  ¬ is_linear_equation_in_two_vars equation_D := 
by 
  sorry

end which_is_linear_l912_91295


namespace negation_of_p_l912_91204

   -- Define the proposition p as an existential quantification
   def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2 * x₀ + 3 > 0

   -- State the theorem that negation of p is a universal quantification
   theorem negation_of_p : ¬ p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 :=
   by sorry
   
end negation_of_p_l912_91204


namespace least_sub_to_make_div_by_10_l912_91298

theorem least_sub_to_make_div_by_10 : 
  ∃ n, n = 8 ∧ ∀ k, 427398 - k = 10 * m → k ≥ n ∧ k = 8 :=
sorry

end least_sub_to_make_div_by_10_l912_91298


namespace parallelogram_area_l912_91292

/-- The area of a parallelogram is given by the product of its base and height. 
Given a parallelogram ABCD with base BC of 4 units and height of 2 units, 
prove its area is 8 square units. --/
theorem parallelogram_area (base height : ℝ) (h_base : base = 4) (h_height : height = 2) : 
  base * height = 8 :=
by
  rw [h_base, h_height]
  norm_num
  done

end parallelogram_area_l912_91292


namespace petya_max_margin_l912_91208

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l912_91208


namespace _l912_91243

-- Define the notion of opposite (additive inverse) of a number
def opposite (n : Int) : Int :=
  -n

-- State the theorem that the opposite of -5 is 5
example : opposite (-5) = 5 := by
  -- Skipping the proof with sorry
  sorry

end _l912_91243


namespace coplanar_iff_m_eq_neg_8_l912_91236

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)
variable (m : ℝ)

theorem coplanar_iff_m_eq_neg_8 
  (h : 4 • A - 3 • B + 7 • C + m • D = 0) : m = -8 ↔ ∃ a b c d : ℝ, a + b + c + d = 0 ∧ a • A + b • B + c • C + d • D = 0 :=
by
  sorry

end coplanar_iff_m_eq_neg_8_l912_91236


namespace min_f_of_shangmei_number_l912_91289

def is_shangmei_number (a b c d : ℕ) : Prop :=
  a + c = 11 ∧ b + d = 11

def f (a b : ℕ) : ℚ :=
  (b - (11 - b) : ℚ) / (a - (11 - a))

def G (a b : ℕ) : ℤ :=
  20 * a + 2 * b - 121

def is_multiple_of_7 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 7 * k

theorem min_f_of_shangmei_number :
  ∃ (a b c d : ℕ), a < b ∧ is_shangmei_number a b c d ∧ is_multiple_of_7 (G a b) ∧ f a b = -3 :=
sorry

end min_f_of_shangmei_number_l912_91289


namespace median_and_mode_of_successful_shots_l912_91221

theorem median_and_mode_of_successful_shots :
  let shots := [3, 6, 4, 6, 4, 3, 6, 5, 7]
  let sorted_shots := [3, 3, 4, 4, 5, 6, 6, 6, 7]
  let median := sorted_shots[4]  -- 4 is the index for the 5th element (0-based indexing)
  let mode := 6  -- determined by the number that appears most frequently
  median = 5 ∧ mode = 6 :=
by
  sorry

end median_and_mode_of_successful_shots_l912_91221


namespace hypotenuse_length_l912_91217

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l912_91217


namespace unique_rectangles_perimeter_sum_correct_l912_91267

def unique_rectangle_sum_of_perimeters : ℕ :=
  let possible_pairs := [(4, 12), (6, 6)]
  let perimeters := possible_pairs.map (λ (p : ℕ × ℕ) => 2 * (p.1 + p.2))
  perimeters.sum

theorem unique_rectangles_perimeter_sum_correct : unique_rectangle_sum_of_perimeters = 56 :=
  by 
  -- skipping actual proof
  sorry

end unique_rectangles_perimeter_sum_correct_l912_91267


namespace determine_b2050_l912_91227

theorem determine_b2050 (b : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h₁ : b 1 = 3 + Real.sqrt 2)
  (h₂ : b 2021 = 7 + 2 * Real.sqrt 2) :
  b 2050 = (7 - 2 * Real.sqrt 2) / 41 := 
sorry

end determine_b2050_l912_91227


namespace regular_ngon_on_parallel_lines_l912_91207

theorem regular_ngon_on_parallel_lines (n : ℕ) : 
  (∃ f : ℝ → ℝ, (∀ m : ℕ, ∃ k : ℕ, f (m * (360 / n)) = k * (360 / n))) ↔
  n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end regular_ngon_on_parallel_lines_l912_91207


namespace salary_of_b_l912_91250

theorem salary_of_b (S_A S_B : ℝ)
  (h1 : S_A + S_B = 14000)
  (h2 : 0.20 * S_A = 0.15 * S_B) :
  S_B = 8000 :=
by
  sorry

end salary_of_b_l912_91250


namespace length_of_AB_l912_91252

-- Define the distances given as conditions
def AC : ℝ := 5
def BD : ℝ := 6
def CD : ℝ := 3

-- Define the linear relationship of points A, B, C, D on the line
def points_on_line_in_order := true -- This is just a placeholder

-- Main theorem to prove
theorem length_of_AB : AB = 2 :=
by
  -- Apply the conditions and the linear relationships
  have BC : ℝ := BD - CD
  have AB : ℝ := AC - BC
  -- This would contain the actual proof using steps, but we skip it here
  sorry

end length_of_AB_l912_91252


namespace f_diff_l912_91273

def f (n : ℕ) : ℚ := (1 / 3 : ℚ) * n * (n + 1) * (n + 2)

theorem f_diff (r : ℕ) : f r - f (r - 1) = r * (r + 1) := 
by {
  -- proof goes here
  sorry
}

end f_diff_l912_91273


namespace farmer_cows_more_than_goats_l912_91237

-- Definitions of the variables
variables (C P G x : ℕ)

-- Conditions given in the problem
def twice_as_many_pigs_as_cows : Prop := P = 2 * C
def more_cows_than_goats : Prop := C = G + x
def goats_count : Prop := G = 11
def total_animals : Prop := C + P + G = 56

-- The theorem to prove
theorem farmer_cows_more_than_goats
  (h1 : twice_as_many_pigs_as_cows C P)
  (h2 : more_cows_than_goats C G x)
  (h3 : goats_count G)
  (h4 : total_animals C P G) :
  C - G = 4 :=
sorry

end farmer_cows_more_than_goats_l912_91237


namespace max_surface_area_of_cut_l912_91279

noncomputable def max_sum_surface_areas (l w h : ℝ) : ℝ :=
  if l = 5 ∧ w = 4 ∧ h = 3 then 144 else 0

theorem max_surface_area_of_cut (l w h : ℝ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) : 
  max_sum_surface_areas l w h = 144 :=
by 
  rw [max_sum_surface_areas, if_pos]
  exact ⟨h_l, h_w, h_h⟩

end max_surface_area_of_cut_l912_91279


namespace arithmetic_sequence_property_l912_91201

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end arithmetic_sequence_property_l912_91201


namespace correctOptionOnlyC_l912_91233

-- Definitions for the transformations
def isTransformA (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b^2) / (a^2)) 
def isTransformB (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b + 1) / (a + 1))
def isTransformC (a b : ℝ) : Prop := (a ≠ 0) → (b / a = (a * b) / (a^2))
def isTransformD (a b : ℝ) : Prop := (a ≠ 0) → ((-b + 1) / a = -(b + 1) / a)

-- Main theorem to assert the correctness of the transformations
theorem correctOptionOnlyC (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬isTransformA a b ∧ ¬isTransformB a b ∧ isTransformC a b ∧ ¬isTransformD a b :=
by
  sorry

end correctOptionOnlyC_l912_91233


namespace range_of_m_l912_91215

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

theorem range_of_m :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → 2^x - Real.log x / Real.log (1/2) + m ≤ 0) →
  m ≤ -5 :=
sorry

end range_of_m_l912_91215


namespace lily_pad_half_coverage_l912_91211

-- Define the conditions in Lean
def doubles_daily (size: ℕ → ℕ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def covers_entire_lake (size: ℕ → ℕ) (total_size: ℕ) : Prop :=
  size 34 = total_size

-- The main statement to prove
theorem lily_pad_half_coverage (size : ℕ → ℕ) (total_size : ℕ) 
  (h1 : doubles_daily size) 
  (h2 : covers_entire_lake size total_size) : 
  size 33 = total_size / 2 :=
sorry

end lily_pad_half_coverage_l912_91211


namespace men_days_proof_l912_91225

noncomputable def time_to_complete (m d e r : ℕ) : ℕ :=
  (m * d) / (e * (m + r))

theorem men_days_proof (m d e r t : ℕ) (h1 : d = (m * d) / (m * e))
  (h2 : t = (m * d) / (e * (m + r))) :
  t = (m * d) / (e * (m + r)) :=
by
  -- The proof would go here
  sorry

end men_days_proof_l912_91225


namespace new_concentration_of_mixture_l912_91214

theorem new_concentration_of_mixture
  (v1_cap : ℝ) (v1_alcohol_percent : ℝ)
  (v2_cap : ℝ) (v2_alcohol_percent : ℝ)
  (new_vessel_cap : ℝ) (poured_liquid : ℝ)
  (filled_water : ℝ) :
  v1_cap = 2 →
  v1_alcohol_percent = 0.25 →
  v2_cap = 6 →
  v2_alcohol_percent = 0.50 →
  new_vessel_cap = 10 →
  poured_liquid = 8 →
  filled_water = (new_vessel_cap - poured_liquid) →
  ((v1_cap * v1_alcohol_percent + v2_cap * v2_alcohol_percent) / new_vessel_cap) = 0.35 :=
by
  intros v1_h v1_per_h v2_h v2_per_h v_new_h poured_h filled_h
  sorry

end new_concentration_of_mixture_l912_91214


namespace probability_event_in_single_trial_l912_91232

theorem probability_event_in_single_trial (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (1 - p)^4 = 16 / 81) : 
  p = 1 / 3 :=
sorry

end probability_event_in_single_trial_l912_91232


namespace annie_milkshakes_l912_91268

theorem annie_milkshakes
  (A : ℕ) (C_hamburger : ℕ) (C_milkshake : ℕ) (H : ℕ) (L : ℕ)
  (initial_money : A = 120)
  (hamburger_cost : C_hamburger = 4)
  (milkshake_cost : C_milkshake = 3)
  (hamburgers_bought : H = 8)
  (money_left : L = 70) :
  ∃ (M : ℕ), A - H * C_hamburger - M * C_milkshake = L ∧ M = 6 :=
by
  sorry

end annie_milkshakes_l912_91268


namespace minimum_cards_to_draw_to_ensure_2_of_each_suit_l912_91276

noncomputable def min_cards_to_draw {total_cards : ℕ} {suit_count : ℕ} {cards_per_suit : ℕ} {joker_count : ℕ}
  (h_total : total_cards = 54)
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : ℕ :=
  43

theorem minimum_cards_to_draw_to_ensure_2_of_each_suit 
  (total_cards suit_count cards_per_suit joker_count : ℕ)
  (h_total : total_cards = 54) 
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : 
  min_cards_to_draw h_total h_suits h_cards_per_suit h_jokers = 43 :=
  by
  sorry

end minimum_cards_to_draw_to_ensure_2_of_each_suit_l912_91276


namespace units_digit_sum_l912_91255

def base8_to_base10 (n : Nat) : Nat :=
  let units := n % 10
  let tens := (n / 10) % 10
  tens * 8 + units

theorem units_digit_sum (n1 n2 : Nat) (h1 : n1 = 45) (h2 : n2 = 67) : ((base8_to_base10 n1) + (base8_to_base10 n2)) % 8 = 4 := by
  sorry

end units_digit_sum_l912_91255


namespace pots_on_each_shelf_l912_91247

variable (x : ℕ)
variable (h1 : 4 * 3 * x = 60)

theorem pots_on_each_shelf : x = 5 := by
  -- proof will go here
  sorry

end pots_on_each_shelf_l912_91247


namespace candidates_count_l912_91293

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 90) : n = 10 :=
by
  sorry

end candidates_count_l912_91293
