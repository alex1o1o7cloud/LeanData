import Mathlib

namespace parkway_elementary_girls_not_playing_soccer_l3342_334234

theorem parkway_elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (boys_playing_soccer_percentage : ℚ)
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : playing_soccer = 250)
  (h4 : boys_playing_soccer_percentage = 86 / 100)
  : ℕ := by
  sorry

#check parkway_elementary_girls_not_playing_soccer

end parkway_elementary_girls_not_playing_soccer_l3342_334234


namespace train_journey_time_l3342_334237

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4 / 5 * usual_speed) * (usual_time + 1 / 2) = usual_speed * usual_time → 
  usual_time = 2 := by
sorry

end train_journey_time_l3342_334237


namespace sum_of_two_equals_third_l3342_334223

theorem sum_of_two_equals_third (x y z : ℤ) 
  (h1 : x + y = z) (h2 : y + z = x) (h3 : z + x = y) : 
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end sum_of_two_equals_third_l3342_334223


namespace complex_power_sum_l3342_334213

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^500 + 1/(z^500) = 2 * Real.cos (100 * π / 180) := by
  sorry

end complex_power_sum_l3342_334213


namespace min_value_of_sum_l3342_334210

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → (∀ x y : ℝ, a > 0 ∧ b > 0 ∧ x / a + y / b = 1 → a + 4 * b ≥ 9) := by
  sorry

end min_value_of_sum_l3342_334210


namespace cost_change_l3342_334298

theorem cost_change (t : ℝ) (b₁ b₂ : ℝ) (h : t * b₂^4 = 16 * t * b₁^4) :
  b₂ = 2 * b₁ := by
  sorry

end cost_change_l3342_334298


namespace total_crayons_calculation_l3342_334203

/-- Given a group of children where each child has a certain number of crayons,
    calculate the total number of crayons. -/
def total_crayons (crayons_per_child : ℕ) (num_children : ℕ) : ℕ :=
  crayons_per_child * num_children

/-- Theorem stating that the total number of crayons is 648 when each child has 18 crayons
    and there are 36 children. -/
theorem total_crayons_calculation :
  total_crayons 18 36 = 648 := by
  sorry

end total_crayons_calculation_l3342_334203


namespace vector_simplification_l3342_334273

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_simplification (a b : V) : 
  2 • (a + b) - a = a + 2 • b := by sorry

end vector_simplification_l3342_334273


namespace perfect_squares_between_210_and_560_l3342_334292

theorem perfect_squares_between_210_and_560 :
  (Finset.filter (fun n => 210 < n^2 ∧ n^2 < 560) (Finset.range 24)).card = 9 :=
by sorry

end perfect_squares_between_210_and_560_l3342_334292


namespace remaining_apples_l3342_334209

def initial_apples : ℕ := 150

def sold_to_jill (apples : ℕ) : ℕ :=
  apples - (apples * 30 / 100)

def sold_to_june (apples : ℕ) : ℕ :=
  apples - (apples * 20 / 100)

def give_to_teacher (apples : ℕ) : ℕ :=
  apples - 1

theorem remaining_apples :
  give_to_teacher (sold_to_june (sold_to_jill initial_apples)) = 83 := by
  sorry

end remaining_apples_l3342_334209


namespace dice_probability_l3342_334202

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of a single die showing an even number -/
def p_even : ℚ := 1/2

/-- The probability of a single die showing an odd number -/
def p_odd : ℚ := 1/2

/-- The number of dice required to show even numbers -/
def num_even : ℕ := 4

/-- The number of dice required to show odd numbers -/
def num_odd : ℕ := 4

theorem dice_probability : 
  (Nat.choose num_dice num_even : ℚ) * p_even ^ num_dice = 35/128 := by
  sorry

end dice_probability_l3342_334202


namespace gcd_of_powers_minus_one_l3342_334247

theorem gcd_of_powers_minus_one : 
  Nat.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end gcd_of_powers_minus_one_l3342_334247


namespace solution_set_l3342_334235

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- State the theorem
theorem solution_set (hf : StrictMono f) (hd : ∀ x ∈ domain, f x ≠ 0) :
  {x : ℝ | f x > f (8 * (x - 2))} = {x : ℝ | 2 < x ∧ x < 16/7} := by sorry

end solution_set_l3342_334235


namespace warden_citations_l3342_334266

/-- The total number of citations issued by a park warden -/
theorem warden_citations (littering : ℕ) (off_leash : ℕ) (parking : ℕ) 
  (h1 : littering = off_leash)
  (h2 : parking = 2 * littering)
  (h3 : littering = 4) : 
  littering + off_leash + parking = 16 := by
  sorry

end warden_citations_l3342_334266


namespace smallest_positive_a_l3342_334251

-- Define the equation
def equation (x a : ℚ) : Prop :=
  (((x - a) / 2 + (x - 2*a) / 3) / ((x + 4*a) / 5 - (x + 3*a) / 4)) =
  (((x - 3*a) / 4 + (x - 4*a) / 5) / ((x + 2*a) / 3 - (x + a) / 2))

-- Define what it means for the equation to have an integer root
def has_integer_root (a : ℚ) : Prop :=
  ∃ x : ℤ, equation x a

-- State the theorem
theorem smallest_positive_a : 
  (∀ a : ℚ, 0 < a ∧ a < 419/421 → ¬ has_integer_root a) ∧ 
  has_integer_root (419/421) :=
sorry

end smallest_positive_a_l3342_334251


namespace movie_production_profit_l3342_334287

def movie_production (main_actor_fee supporting_actor_fee extra_fee : ℕ)
                     (main_actor_food supporting_actor_food crew_food : ℕ)
                     (post_production_cost revenue : ℕ) : Prop :=
  let num_main_actors : ℕ := 2
  let num_supporting_actors : ℕ := 3
  let num_extra : ℕ := 1
  let total_people : ℕ := 50
  let actor_fees := num_main_actors * main_actor_fee + 
                    num_supporting_actors * supporting_actor_fee + 
                    num_extra * extra_fee
  let food_cost := num_main_actors * main_actor_food + 
                   (num_supporting_actors + num_extra) * supporting_actor_food + 
                   (total_people - num_main_actors - num_supporting_actors - num_extra) * crew_food
  let equipment_cost := 2 * (actor_fees + food_cost)
  let total_cost := actor_fees + food_cost + equipment_cost + post_production_cost
  let profit := revenue - total_cost
  profit = 4584

theorem movie_production_profit :
  movie_production 500 100 50 10 5 3 850 10000 := by
  sorry

end movie_production_profit_l3342_334287


namespace limit_of_a_is_2_l3342_334226

def a (n : ℕ) : ℚ := (4 * n - 3) / (2 * n + 1)

theorem limit_of_a_is_2 : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε := by
  sorry

end limit_of_a_is_2_l3342_334226


namespace student_scores_l3342_334207

theorem student_scores (math physics chemistry : ℕ) : 
  math + physics = 32 →
  (math + chemistry) / 2 = 26 →
  ∃ x : ℕ, chemistry = physics + x ∧ x = 20 := by
sorry

end student_scores_l3342_334207


namespace smallest_sum_of_squares_l3342_334214

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 221) :
  x^2 + y^2 ≥ 229 :=
by sorry

end smallest_sum_of_squares_l3342_334214


namespace divisibility_condition_l3342_334233

theorem divisibility_condition (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end divisibility_condition_l3342_334233


namespace product_of_sum_of_roots_l3342_334296

theorem product_of_sum_of_roots (x : ℝ) :
  (Real.sqrt (8 + x) + Real.sqrt (15 - x) = 6) →
  (8 + x) * (15 - x) = 169 / 4 := by
  sorry

end product_of_sum_of_roots_l3342_334296


namespace school_sections_l3342_334279

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) :
  let section_size := Nat.gcd boys girls
  let boy_sections := boys / section_size
  let girl_sections := girls / section_size
  boy_sections + girl_sections = 29 := by
sorry

end school_sections_l3342_334279


namespace factorization_of_polynomial_l3342_334270

theorem factorization_of_polynomial (z : ℝ) : 
  75 * z^24 + 225 * z^48 = 75 * z^24 * (1 + 3 * z^24) := by
sorry

end factorization_of_polynomial_l3342_334270


namespace count_ways_2016_l3342_334288

/-- The number of ways to write 2016 as the sum of twos and threes, ignoring order -/
def ways_to_write_2016 : ℕ :=
  (Finset.range 337).card

/-- The theorem stating that there are 337 ways to write 2016 as the sum of twos and threes -/
theorem count_ways_2016 : ways_to_write_2016 = 337 := by
  sorry

end count_ways_2016_l3342_334288


namespace w_value_l3342_334230

def cubic_poly (x : ℝ) := x^3 - 4*x^2 + 2*x + 1

def second_poly (x u v w : ℝ) := x^3 + u*x^2 + v*x + w

theorem w_value (p q r u v w : ℝ) :
  cubic_poly p = 0 ∧ cubic_poly q = 0 ∧ cubic_poly r = 0 →
  second_poly (p + q) u v w = 0 ∧ second_poly (q + r) u v w = 0 ∧ second_poly (r + p) u v w = 0 →
  w = 15 := by sorry

end w_value_l3342_334230


namespace alexandra_magazines_l3342_334242

theorem alexandra_magazines : 
  let friday_magazines : ℕ := 8
  let saturday_magazines : ℕ := 12
  let sunday_magazines : ℕ := 4 * friday_magazines
  let chewed_magazines : ℕ := 4
  friday_magazines + saturday_magazines + sunday_magazines - chewed_magazines = 48
  := by sorry

end alexandra_magazines_l3342_334242


namespace triangle_angle_proof_l3342_334264

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the problem
theorem triangle_angle_proof (t : Triangle) (m n : ℝ × ℝ) :
  m = (t.a + t.c, -t.b) →
  n = (t.a - t.c, t.b) →
  m.1 * n.1 + m.2 * n.2 = t.b * t.c →
  0 < t.A →
  t.A < π →
  t.A = 2 * π / 3 := by
  sorry

end triangle_angle_proof_l3342_334264


namespace enlarged_poster_height_l3342_334239

/-- Calculates the new height of a proportionally enlarged poster -/
def new_poster_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem: The new height of the enlarged poster is 10 inches -/
theorem enlarged_poster_height :
  new_poster_height 3 2 15 = 10 := by
  sorry

end enlarged_poster_height_l3342_334239


namespace seeds_per_flowerbed_l3342_334211

theorem seeds_per_flowerbed 
  (total_seeds : ℕ) 
  (num_flowerbeds : ℕ) 
  (h1 : total_seeds = 45) 
  (h2 : num_flowerbeds = 9) 
  (h3 : total_seeds % num_flowerbeds = 0) :
  total_seeds / num_flowerbeds = 5 := by
sorry

end seeds_per_flowerbed_l3342_334211


namespace hcf_of_8_and_12_l3342_334276

theorem hcf_of_8_and_12 :
  let a : ℕ := 8
  let b : ℕ := 12
  Nat.lcm a b = 24 →
  Nat.gcd a b = 4 :=
by
  sorry

end hcf_of_8_and_12_l3342_334276


namespace largest_divisor_of_consecutive_integers_with_five_l3342_334259

def is_divisible (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem largest_divisor_of_consecutive_integers_with_five (n : ℤ) :
  (is_divisible n 5 ∨ is_divisible (n + 1) 5 ∨ is_divisible (n + 2) 5) →
  is_divisible (n * (n + 1) * (n + 2)) 15 ∧
  ∀ m : ℤ, m > 15 → ¬(∀ k : ℤ, (is_divisible k 5 ∨ is_divisible (k + 1) 5 ∨ is_divisible (k + 2) 5) →
                              is_divisible (k * (k + 1) * (k + 2)) m) :=
by sorry

end largest_divisor_of_consecutive_integers_with_five_l3342_334259


namespace tangent_perpendicular_condition_l3342_334240

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0
def circle2 (a x y : ℝ) : Prop := x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0

-- Define the condition for perpendicular tangent lines
def perpendicular_tangents (a m n : ℝ) : Prop :=
  (n + 2) / m * (n + 1) / (m - (1 - a)) = -1

-- Define the theorem
theorem tangent_perpendicular_condition :
  ∃ (a : ℝ), a = -2 ∧
  ∀ (m n : ℝ), circle1 m n → circle2 a m n → perpendicular_tangents a m n :=
sorry

end tangent_perpendicular_condition_l3342_334240


namespace bench_cost_l3342_334216

theorem bench_cost (total_cost bench_cost table_cost : ℝ) : 
  total_cost = 450 →
  table_cost = 2 * bench_cost →
  total_cost = bench_cost + table_cost →
  bench_cost = 150 := by
sorry

end bench_cost_l3342_334216


namespace cyclists_initial_distance_l3342_334250

/-- The initial distance between two cyclists -/
def initial_distance : ℝ := 50

/-- The speed of each cyclist -/
def cyclist_speed : ℝ := 10

/-- The speed of the fly -/
def fly_speed : ℝ := 15

/-- The total distance covered by the fly -/
def fly_distance : ℝ := 37.5

/-- Theorem stating that the initial distance between the cyclists is 50 miles -/
theorem cyclists_initial_distance :
  initial_distance = 
    (2 * cyclist_speed * fly_distance) / fly_speed :=
by sorry

end cyclists_initial_distance_l3342_334250


namespace test_retake_count_l3342_334249

theorem test_retake_count (total : ℕ) (passed : ℕ) (retake : ℕ) : 
  total = 2500 → passed = 375 → retake = total - passed → retake = 2125 := by
  sorry

end test_retake_count_l3342_334249


namespace qi_winning_probability_l3342_334295

-- Define the horse strengths
structure HorseStrengths where
  tian_top_better_than_qi_middle : Prop
  tian_top_worse_than_qi_top : Prop
  tian_middle_better_than_qi_bottom : Prop
  tian_middle_worse_than_qi_middle : Prop
  tian_bottom_worse_than_qi_bottom : Prop

-- Define the probability of Qi's horse winning
def probability_qi_wins (strengths : HorseStrengths) : ℚ := 2/3

-- Theorem statement
theorem qi_winning_probability (strengths : HorseStrengths) :
  probability_qi_wins strengths = 2/3 := by sorry

end qi_winning_probability_l3342_334295


namespace roots_of_transformed_equation_l3342_334236

theorem roots_of_transformed_equation
  (p q : ℝ) (x₁ x₂ : ℝ)
  (h1 : x₁^2 + p*x₁ + q = 0)
  (h2 : x₂^2 + p*x₂ + q = 0)
  : (-x₁)^2 - p*(-x₁) + q = 0 ∧ (-x₂)^2 - p*(-x₂) + q = 0 :=
by sorry

end roots_of_transformed_equation_l3342_334236


namespace system_solution_l3342_334285

theorem system_solution : ∃ (a b c d : ℝ), 
  (a + c = -4 ∧ 
   a * c + b + d = 6 ∧ 
   a * d + b * c = -5 ∧ 
   b * d = 2) ∧
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) := by
  sorry

end system_solution_l3342_334285


namespace total_players_l3342_334299

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ)
  (h1 : kabadi = 10)
  (h2 : kho_kho_only = 30)
  (h3 : both = 5) :
  kabadi - both + kho_kho_only = 40 := by
  sorry

end total_players_l3342_334299


namespace trader_gain_percentage_l3342_334253

theorem trader_gain_percentage (cost : ℝ) (h : cost > 0) : 
  (22 * cost) / (88 * cost) * 100 = 25 := by
  sorry

end trader_gain_percentage_l3342_334253


namespace sum_of_coefficients_l3342_334205

def sequence_u (n : ℕ) : ℝ :=
  sorry

theorem sum_of_coefficients :
  (∃ (a b c : ℝ), ∀ (n : ℕ), sequence_u n = a * n^2 + b * n + c) →
  (sequence_u 1 = 7) →
  (∀ (n : ℕ), sequence_u (n + 1) - sequence_u n = 5 + 3 * (n - 1)) →
  (∃ (a b c : ℝ), 
    (∀ (n : ℕ), sequence_u n = a * n^2 + b * n + c) ∧
    (a + b + c = 7)) :=
by sorry

end sum_of_coefficients_l3342_334205


namespace tan_240_plus_sin_neg_420_l3342_334290

theorem tan_240_plus_sin_neg_420 :
  Real.tan (240 * π / 180) + Real.sin ((-420) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end tan_240_plus_sin_neg_420_l3342_334290


namespace two_numbers_difference_l3342_334271

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : 
  |x - y| = 2 := by
sorry

end two_numbers_difference_l3342_334271


namespace binary_110110_is_54_l3342_334262

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_110110_is_54 :
  binary_to_decimal [true, true, false, true, true, false] = 54 := by
  sorry

end binary_110110_is_54_l3342_334262


namespace f_upper_bound_f_negative_l3342_334252

/-- The function f(x) = ax^2 - (a+1)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

/-- Theorem stating the range of a for which f(x) ≤ 2 for all x in ℝ -/
theorem f_upper_bound (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ -3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2 :=
sorry

/-- Theorem describing the solution set of f(x) < 0 for different ranges of a -/
theorem f_negative (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨
     (a > 1 ∧ 1/a < x ∧ x < 1) ∨
     (a < 0 ∧ ((x < 1/a) ∨ (x > 1))))) :=
sorry

end f_upper_bound_f_negative_l3342_334252


namespace equation_solution_l3342_334258

/-- Given an equation x^4 - 10x^3 - 2(a-11)x^2 + 2(5a+6)x + 2a + a^2 = 0,
    where a is a constant and a ≥ -6, prove the solutions for a and x. -/
theorem equation_solution (a x : ℝ) (h : a ≥ -6) :
  x^4 - 10*x^3 - 2*(a-11)*x^2 + 2*(5*a+6)*x + 2*a + a^2 = 0 →
  ((a = x^2 - 4*x - 2) ∨ (a = x^2 - 6*x)) ∧
  ((∃ (i : Fin 2), x = 2 + (-1)^(i : ℕ) * Real.sqrt (a + 6)) ∨
   (∃ (i : Fin 2), x = 3 + (-1)^(i : ℕ) * Real.sqrt (a + 9))) :=
by sorry

end equation_solution_l3342_334258


namespace solve_linear_equation_l3342_334265

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end solve_linear_equation_l3342_334265


namespace fourth_root_equivalence_l3342_334238

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/2))^(1/4) = x^(5/8) := by
sorry

end fourth_root_equivalence_l3342_334238


namespace f_value_theorem_l3342_334220

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_value_theorem (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f)
  (h_def : ∀ x, 0 < x → x < 1 → f x = 1 / x) :
  f (-5/2) + f 0 = -2 := by
  sorry

end f_value_theorem_l3342_334220


namespace moles_of_CH3Cl_formed_l3342_334293

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the available moles
def available_CH4 : ℝ := 1
def available_Cl2 : ℝ := 1

-- Define the reaction
def methane_chlorine_reaction : Reaction :=
  { reactant1 := "CH4"
  , reactant2 := "Cl2"
  , product1 := "CH3Cl"
  , product2 := "HCl" }

-- Theorem statement
theorem moles_of_CH3Cl_formed (reaction : Reaction) 
  (h1 : reaction = methane_chlorine_reaction)
  (h2 : available_CH4 = 1)
  (h3 : available_Cl2 = 1) :
  ∃ (moles_CH3Cl : ℝ), moles_CH3Cl = 1 :=
sorry

end moles_of_CH3Cl_formed_l3342_334293


namespace train_speed_l3342_334255

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 700) (h2 : time = 20) :
  length / time = 35 := by
  sorry

end train_speed_l3342_334255


namespace intersection_equality_implies_a_equals_one_l3342_334232

theorem intersection_equality_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {1, 2, 5}
  let B : Set ℝ := {a + 4, a}
  A ∩ B = B → a = 1 := by
sorry

end intersection_equality_implies_a_equals_one_l3342_334232


namespace arithmetic_sequence_decreasing_l3342_334204

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_decreasing
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : (a 2 - 1)^3 + 2012 * (a 2 - 1) = 1)
  (h_a2011 : (a 2011 - 1)^3 + 2012 * (a 2011 - 1) = -1) :
  a 2011 < a 2 := by
sorry

end arithmetic_sequence_decreasing_l3342_334204


namespace ratio_x_to_y_l3342_334244

theorem ratio_x_to_y (x y : ℝ) (h : 0.8 * x = 0.2 * y) : x / y = 1 / 4 := by
  sorry

end ratio_x_to_y_l3342_334244


namespace roots_sum_powers_l3342_334206

theorem roots_sum_powers (a b : ℝ) : 
  a + b = 6 → ab = 8 → a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 := by
  sorry

end roots_sum_powers_l3342_334206


namespace gcd_factorial_problem_l3342_334261

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 2520 := by
  sorry

end gcd_factorial_problem_l3342_334261


namespace wilsons_theorem_l3342_334217

theorem wilsons_theorem (p : ℕ) (hp : Prime p) :
  (((Nat.factorial (p - 1)) : ℤ) % p = -1) ∧
  (p^2 ∣ ((Nat.factorial (p - 1)) + 1)) := by
  sorry

end wilsons_theorem_l3342_334217


namespace first_player_can_force_draw_l3342_334218

/-- Represents the state of a square on the game board -/
inductive Square
| Empty : Square
| A : Square
| B : Square

/-- Represents the game board as a list of squares -/
def Board := List Square

/-- Checks if a given board contains the winning sequence ABA -/
def hasWinningSequence (board : Board) : Bool :=
  sorry

/-- Represents a player's move -/
structure Move where
  position : Nat
  letter : Square

/-- Applies a move to the board -/
def applyMove (board : Board) (move : Move) : Board :=
  sorry

/-- Checks if a move is valid on the given board -/
def isValidMove (board : Board) (move : Move) : Bool :=
  sorry

/-- Represents the game state -/
structure GameState where
  board : Board
  currentPlayer : Bool  -- True for first player, False for second player

/-- The main theorem stating that the first player can force a draw -/
theorem first_player_can_force_draw :
  ∃ (strategy : GameState → Move),
    ∀ (game : GameState),
      game.board.length = 14 →
      game.currentPlayer = true →
      ¬(hasWinningSequence (applyMove game.board (strategy game))) :=
sorry

end first_player_can_force_draw_l3342_334218


namespace abc_solution_l3342_334275

/-- Converts a base 7 number to its decimal representation -/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to its base 7 representation -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Represents a two-digit number in base 7 -/
def twoDigitBase7 (tens : ℕ) (ones : ℕ) : ℕ := 7 * tens + ones

/-- Represents a three-digit number in base 7 -/
def threeDigitBase7 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ := 49 * hundreds + 7 * tens + ones

theorem abc_solution (A B C : ℕ) : 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) →  -- non-zero digits
  (A < 7 ∧ B < 7 ∧ C < 7) →  -- less than 7
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) →  -- distinct digits
  (twoDigitBase7 A B + C = twoDigitBase7 C 0) →  -- AB₇ + C₇ = C0₇
  (twoDigitBase7 A B + twoDigitBase7 B A = twoDigitBase7 C C) →  -- AB₇ + BA₇ = CC₇
  threeDigitBase7 A B C = 643  -- ABC = 643 in base 7
  := by sorry


end abc_solution_l3342_334275


namespace colored_isosceles_triangle_l3342_334246

/-- A regular polygon with 4n + 1 vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (4 * n + 1) → ℝ × ℝ

/-- A coloring of 2n vertices in a (4n + 1)-gon -/
def Coloring (n : ℕ) := Fin (4 * n + 1) → Bool

/-- Three vertices form an isosceles triangle -/
def IsIsosceles (p : RegularPolygon n) (v1 v2 v3 : Fin (4 * n + 1)) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- Main theorem -/
theorem colored_isosceles_triangle (n : ℕ) (h : n ≥ 3) (p : RegularPolygon n) (c : Coloring n) :
  ∃ v1 v2 v3 : Fin (4 * n + 1), c v1 ∧ c v2 ∧ c v3 ∧ IsIsosceles p v1 v2 v3 :=
sorry


end colored_isosceles_triangle_l3342_334246


namespace curve_constants_sum_l3342_334260

/-- Given a curve y = ax² + b/x passing through the point (2, -5) with a tangent at this point
    parallel to the line 7x + 2y + 3 = 0, prove that a + b = -43/20 -/
theorem curve_constants_sum (a b : ℝ) : 
  (4 * a + b / 2 = -5) →  -- Curve passes through (2, -5)
  (4 * a - b / 4 = -7/2) →  -- Tangent at (2, -5) is parallel to 7x + 2y + 3 = 0
  a + b = -43/20 := by
  sorry

end curve_constants_sum_l3342_334260


namespace linear_function_composition_l3342_334267

-- Define a linear function
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  IsLinearFunction f → (∀ x, f (f x) = 4 * x + 8) →
  (∀ x, f x = 2 * x + 8 / 3) ∨ (∀ x, f x = -2 * x - 8) :=
by
  sorry

end linear_function_composition_l3342_334267


namespace equation_solutions_l3342_334289

theorem equation_solutions (x y z v : ℤ) : 
  (x^2 + y^2 + z^2 = 2*x*y*z ↔ x = 0 ∧ y = 0 ∧ z = 0) ∧
  (x^2 + y^2 + z^2 + v^2 = 2*x*y*z*v ↔ x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) := by
  sorry

end equation_solutions_l3342_334289


namespace expand_expression_l3342_334212

theorem expand_expression (x : ℝ) : (16*x + 18 - 4*x^2) * 3*x = -12*x^3 + 48*x^2 + 54*x := by
  sorry

end expand_expression_l3342_334212


namespace a2_value_l3342_334263

theorem a2_value (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, x^4 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3 + a₄*(x-2)^4) →
  a₂ = 24 := by
sorry

end a2_value_l3342_334263


namespace intersecting_lines_theorem_l3342_334229

/-- Given two lines that intersect at (-7, 9), prove that the line passing through their coefficients as points has the equation -7x + 9y = 1 -/
theorem intersecting_lines_theorem (A₁ B₁ A₂ B₂ : ℝ) : 
  (A₁ * (-7) + B₁ * 9 = 1) →  -- First line passes through (-7, 9)
  (A₂ * (-7) + B₂ * 9 = 1) →  -- Second line passes through (-7, 9)
  ∃ (k : ℝ), k * (-7) * (A₂ - A₁) = 9 * (B₂ - B₁) ∧   -- Points (A₁, B₁) and (A₂, B₂) satisfy -7x + 9y = k
             k = 1 :=
by sorry

end intersecting_lines_theorem_l3342_334229


namespace arithmetic_sequence_third_term_l3342_334294

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_third_term (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 2 → a 5 + a 7 = 2 * a 4 + 4 → a 3 = 4 := by
  sorry

end arithmetic_sequence_third_term_l3342_334294


namespace alien_number_conversion_l3342_334225

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base-6 representation of the number --/
def alienNumber : List Nat := [4, 5, 1, 2]

theorem alien_number_conversion :
  base6ToBase10 alienNumber = 502 := by
  sorry

#eval base6ToBase10 alienNumber

end alien_number_conversion_l3342_334225


namespace units_digit_of_composite_product_l3342_334268

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_composite_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end units_digit_of_composite_product_l3342_334268


namespace terminal_side_first_quadrant_l3342_334297

-- Define the angle in degrees
def angle : ℤ := -685

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (a : ℤ) : ℤ :=
  (a % 360 + 360) % 360

-- Define a function to determine the quadrant of an angle
def quadrant (a : ℤ) : ℕ :=
  let normalizedAngle := normalizeAngle a
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem terminal_side_first_quadrant :
  quadrant angle = 1 := by sorry

end terminal_side_first_quadrant_l3342_334297


namespace foil_covered_prism_width_l3342_334278

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the inner dimensions of the prism (not covered by foil) -/
def inner_prism : PrismDimensions :=
  { length := 4,
    width := 8,
    height := 4 }

/-- Represents the outer dimensions of the prism (covered by foil) -/
def outer_prism : PrismDimensions :=
  { length := inner_prism.length + 2,
    width := inner_prism.width + 2,
    height := inner_prism.height + 2 }

/-- The main theorem to prove -/
theorem foil_covered_prism_width :
  (volume inner_prism = 128) →
  (inner_prism.width = 2 * inner_prism.length) →
  (inner_prism.width = 2 * inner_prism.height) →
  (outer_prism.width = 10) := by
  sorry

end foil_covered_prism_width_l3342_334278


namespace simplify_sqrt_difference_l3342_334219

theorem simplify_sqrt_difference (x : ℝ) (h : x ≤ 2) : 
  Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9) = -1 :=
by sorry

end simplify_sqrt_difference_l3342_334219


namespace square_root_sum_implies_product_l3342_334201

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (25 - x) = 9) →
  ((10 + x) * (25 - x) = 529) := by
  sorry

end square_root_sum_implies_product_l3342_334201


namespace return_trip_time_l3342_334221

/-- Given a route with uphill and downhill sections, prove the return trip time -/
theorem return_trip_time (total_distance : ℝ) (uphill_speed downhill_speed : ℝ) 
  (time_ab : ℝ) (h1 : total_distance = 21) (h2 : uphill_speed = 4) 
  (h3 : downhill_speed = 6) (h4 : time_ab = 4.25) : ∃ (time_ba : ℝ), time_ba = 4.5 := by
  sorry

end return_trip_time_l3342_334221


namespace rectangles_on_4x4_grid_l3342_334227

/-- The number of rectangles that can be formed on a 4x4 grid --/
def num_rectangles_4x4 : ℕ := 36

/-- The size of the grid --/
def grid_size : ℕ := 4

/-- Theorem: The number of rectangles on a 4x4 grid is 36 --/
theorem rectangles_on_4x4_grid :
  num_rectangles_4x4 = (grid_size.choose 2) * (grid_size.choose 2) :=
by sorry

end rectangles_on_4x4_grid_l3342_334227


namespace sticker_collection_l3342_334222

theorem sticker_collection (karl_stickers : ℕ) : 
  (∃ (ryan_stickers ben_stickers : ℕ),
    ryan_stickers = karl_stickers + 20 ∧
    ben_stickers = ryan_stickers - 10 ∧
    karl_stickers + ryan_stickers + ben_stickers = 105) →
  karl_stickers = 25 := by
sorry

end sticker_collection_l3342_334222


namespace max_length_complex_l3342_334257

theorem max_length_complex (ω : ℂ) (h : Complex.abs ω = 1) :
  ∃ (max : ℝ), max = 108 ∧ ∀ (z : ℂ), Complex.abs ((ω + 2)^3 * (ω - 3)^2) ≤ max :=
sorry

end max_length_complex_l3342_334257


namespace chord_segment_lengths_l3342_334245

theorem chord_segment_lengths (R : ℝ) (OM : ℝ) (AB : ℝ) (AM MB : ℝ) : 
  R = 15 →
  OM = 13 →
  AB = 18 →
  AM + MB = AB →
  OM^2 = R^2 - (AB/2)^2 + ((AM - MB)/2)^2 →
  AM = 14 ∧ MB = 4 :=
by sorry

end chord_segment_lengths_l3342_334245


namespace hyperbola_equation_l3342_334254

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    focal length 2√6, and an asymptote l such that the distance from (1,0) to l is √6/3,
    prove that the equation of the hyperbola is x²/2 - y²/4 = 1. -/
theorem hyperbola_equation (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_focal : Real.sqrt (a^2 + b^2) = Real.sqrt 6)
  (h_asymptote : ∃ (k : ℝ), k * a = b ∧ k * b = a)
  (h_distance : (b / Real.sqrt (a^2 + b^2)) = Real.sqrt 6 / 3) :
  a^2 = 2 ∧ b^2 = 4 := by
  sorry

end hyperbola_equation_l3342_334254


namespace marbles_left_l3342_334224

def initial_marbles : ℕ := 38
def lost_marbles : ℕ := 15

theorem marbles_left : initial_marbles - lost_marbles = 23 := by
  sorry

end marbles_left_l3342_334224


namespace max_intersections_theorem_l3342_334291

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : sides ≥ 3

/-- Represents the configuration of two convex polygons -/
structure TwoPolygons where
  P₁ : ConvexPolygon
  P₂ : ConvexPolygon
  sameplane : True  -- Represents that P₁ and P₂ are on the same plane
  no_overlap : True  -- Represents that P₁ and P₂ do not have overlapping line segments
  size_order : P₁.sides ≤ P₂.sides

/-- The function that calculates the maximum number of intersection points -/
def max_intersections (tp : TwoPolygons) : ℕ := 2 * tp.P₁.sides

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersections_theorem (tp : TwoPolygons) : 
  max_intersections tp = 2 * tp.P₁.sides := by sorry

end max_intersections_theorem_l3342_334291


namespace monotonic_increasing_interval_l3342_334282

/-- The function f(x) = x^2 / 2^x is monotonically increasing on the interval (0, 2/ln(2)) -/
theorem monotonic_increasing_interval (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 / 2^x) →
  (∃ a b, a = 0 ∧ b = 2 / Real.log 2 ∧
    ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) := by
  sorry

end monotonic_increasing_interval_l3342_334282


namespace solution_set_when_a_eq_2_range_of_a_for_inequality_l3342_334208

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem range_of_a_for_inequality :
  (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 := by sorry

end solution_set_when_a_eq_2_range_of_a_for_inequality_l3342_334208


namespace similar_triangles_leg_sum_l3342_334215

theorem similar_triangles_leg_sum (a b c d : ℕ) : 
  a * b = 18 →  -- area of smaller triangle is 9
  a^2 + b^2 = 25 →  -- hypotenuse of smaller triangle is 5
  a ≠ 3 ∨ b ≠ 4 →  -- not a 3-4-5 triangle
  c * d = 450 →  -- area of larger triangle is 225
  (c : ℝ) / a = (d : ℝ) / b →  -- triangles are similar
  (c + d : ℝ) = 45 := by
  sorry

end similar_triangles_leg_sum_l3342_334215


namespace peach_difference_l3342_334280

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 12 →
  jake_peaches + 1 = jill_peaches →
  steven_peaches = jake_peaches + 16 →
  steven_peaches - jill_peaches = 15 := by
  sorry

end peach_difference_l3342_334280


namespace pressure_valve_problem_l3342_334286

/-- Represents the constant ratio between pressure change and temperature -/
def k : ℚ := (5 * 4 - 6) / (10 + 20)

/-- The pressure-temperature relationship function -/
def pressure_temp_relation (x t : ℚ) : Prop :=
  (5 * x - 6) / (t + 20) = k

theorem pressure_valve_problem :
  pressure_temp_relation 4 10 →
  pressure_temp_relation (34/5) 40 :=
by sorry

end pressure_valve_problem_l3342_334286


namespace expansion_properties_l3342_334241

/-- Given the expansion of (2x + 3/∛x)^n, where the ratio of the binomial coefficient
    of the third term to that of the second term is 5:2, prove the following: -/
theorem expansion_properties (n : ℕ) (x : ℝ) :
  (Nat.choose n 2 : ℚ) / (Nat.choose n 1 : ℚ) = 5 / 2 →
  (n = 6 ∧
   (∃ (r : ℕ), Nat.choose 6 r * 2^(6-r) * 3^r * x^(6 - 4/3*r) = 4320 * x^2) ∧
   (∃ (k : ℕ), Nat.choose 6 k * 2^(6-k) * 3^k * x^((2:ℝ)/3) = 4860 * x^((2:ℝ)/3) ∧
               ∀ (j : ℕ), j ≠ k → Nat.choose 6 j * 2^(6-j) * 3^j ≤ Nat.choose 6 k * 2^(6-k) * 3^k)) :=
by sorry

end expansion_properties_l3342_334241


namespace two_thirds_plus_six_l3342_334200

theorem two_thirds_plus_six (x : ℝ) : x = 6 → (2 / 3 * x) + 6 = 10 := by
  sorry

end two_thirds_plus_six_l3342_334200


namespace total_weight_sold_l3342_334243

/-- Calculates the total weight of bags sold in a day given the sales data and weight per bag -/
theorem total_weight_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions
  morning_carrots afternoon_carrots potato_weight onion_weight carrot_weight : ℕ) :
  morning_potatoes = 29 →
  afternoon_potatoes = 17 →
  morning_onions = 15 →
  afternoon_onions = 22 →
  morning_carrots = 12 →
  afternoon_carrots = 9 →
  potato_weight = 7 →
  onion_weight = 5 →
  carrot_weight = 4 →
  (morning_potatoes + afternoon_potatoes) * potato_weight +
  (morning_onions + afternoon_onions) * onion_weight +
  (morning_carrots + afternoon_carrots) * carrot_weight = 591 :=
by
  sorry

end total_weight_sold_l3342_334243


namespace book_distribution_l3342_334256

def number_of_books : ℕ := 6
def number_of_people : ℕ := 3

def distribute_evenly (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose (n - 2) 2

def distribute_fixed (n : ℕ) : ℕ :=
  Nat.choose n 1 * Nat.choose (n - 1) 2

def distribute_variable (n m : ℕ) : ℕ :=
  Nat.choose n 1 * Nat.choose (n - 1) 2 * Nat.factorial m

theorem book_distribution :
  (distribute_evenly number_of_books number_of_people = 90) ∧
  (distribute_fixed number_of_books = 60) ∧
  (distribute_variable number_of_books number_of_people = 360) := by
  sorry

end book_distribution_l3342_334256


namespace hainan_scientific_notation_l3342_334277

theorem hainan_scientific_notation :
  48500000 = 4.85 * (10 ^ 7) := by
  sorry

end hainan_scientific_notation_l3342_334277


namespace existence_of_z_l3342_334248

theorem existence_of_z (a p x y : ℕ) (hp : Prime p) (hx : x > 0) (hy : y > 0) (ha : a > 0)
  (hx41 : ∃ n : ℕ, x^41 = a + n*p) (hy49 : ∃ n : ℕ, y^49 = a + n*p) :
  ∃ (z : ℕ), z > 0 ∧ ∃ (n : ℕ), z^2009 = a + n*p :=
by sorry

end existence_of_z_l3342_334248


namespace solution_of_system_l3342_334269

def augmented_matrix : Matrix (Fin 2) (Fin 3) ℝ := !![1, -1, 1; 1, 1, 3]

theorem solution_of_system (x y : ℝ) : 
  x = 2 ∧ y = 1 ↔ 
  (augmented_matrix 0 0 * x + augmented_matrix 0 1 * y = augmented_matrix 0 2) ∧
  (augmented_matrix 1 0 * x + augmented_matrix 1 1 * y = augmented_matrix 1 2) :=
by sorry

end solution_of_system_l3342_334269


namespace total_streets_patrolled_in_one_hour_l3342_334284

/-- Represents the patrol rate of a police officer -/
structure PatrolRate where
  streets : ℕ
  hours : ℕ

/-- Calculates the number of streets patrolled per hour -/
def streetsPerHour (rate : PatrolRate) : ℚ :=
  rate.streets / rate.hours

/-- The patrol rates of three officers -/
def officerA : PatrolRate := { streets := 36, hours := 4 }
def officerB : PatrolRate := { streets := 55, hours := 5 }
def officerC : PatrolRate := { streets := 42, hours := 6 }

/-- The total number of streets patrolled by all three officers in one hour -/
def totalStreetsPerHour : ℚ :=
  streetsPerHour officerA + streetsPerHour officerB + streetsPerHour officerC

theorem total_streets_patrolled_in_one_hour :
  totalStreetsPerHour = 27 := by
  sorry

end total_streets_patrolled_in_one_hour_l3342_334284


namespace x_value_proof_l3342_334231

theorem x_value_proof (x : ℝ) 
  (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3)*Complex.I) : 
  x = 3 := by
sorry

end x_value_proof_l3342_334231


namespace exactly_two_true_with_converse_l3342_334274

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := (a / b < 1) → (a < b)
def proposition2 (sides : Fin 4 → ℝ) : Prop := ∀ i j, sides i = sides j
def proposition3 (angles : Fin 3 → ℝ) : Prop := angles 0 = angles 1

-- Define the converses
def converse1 (a b : ℝ) : Prop := (a < b) → (a / b < 1)
def converse2 (sides : Fin 4 → ℝ) : Prop := (∀ i j, sides i = sides j) → (∃ r : ℝ, ∀ i, sides i = r)
def converse3 (angles : Fin 3 → ℝ) : Prop := (angles 0 = angles 1) → (∃ s : ℝ, angles 0 = s ∧ angles 1 = s)

-- Theorem statement
theorem exactly_two_true_with_converse :
  ∃! n : ℕ, n = 2 ∧
  (∀ a b : ℝ, proposition1 a b ∧ converse1 a b) ∨
  (∀ sides : Fin 4 → ℝ, proposition2 sides ∧ converse2 sides) ∨
  (∀ angles : Fin 3 → ℝ, proposition3 angles ∧ converse3 angles) :=
sorry

end exactly_two_true_with_converse_l3342_334274


namespace boys_combined_average_l3342_334228

/-- Represents a high school with average scores for boys, girls, and combined --/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Theorem stating that given the conditions, the average score for boys across two schools is 70.8 --/
theorem boys_combined_average (chs dhs : School)
  (h_chs_boys : chs.boys_avg = 68)
  (h_chs_girls : chs.girls_avg = 73)
  (h_chs_combined : chs.combined_avg = 70)
  (h_dhs_boys : dhs.boys_avg = 75)
  (h_dhs_girls : dhs.girls_avg = 85)
  (h_dhs_combined : dhs.combined_avg = 80) :
  ∃ (c d : ℝ), c > 0 ∧ d > 0 ∧
  (c * chs.boys_avg + d * dhs.boys_avg) / (c + d) = 70.8 := by
  sorry


end boys_combined_average_l3342_334228


namespace parabola_equation_l3342_334283

/-- Given a parabola y = 2px (p > 0) and a point M on it with abscissa 3,
    if |MF| = 2p, then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) :
  ∃ (M : ℝ × ℝ),
    M.1 = 3 ∧
    M.2 = 2 * p * M.1 ∧
    |M.1 - (-p/2)| + M.2 = 2 * p →
  ∀ (x y : ℝ), y = 2 * p * x ↔ y^2 = 4 * x :=
by sorry

end parabola_equation_l3342_334283


namespace ellipse_iff_range_l3342_334272

/-- The equation of an ellipse with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 - m) + y^2 / (m + 1) = 1

/-- The range of m for which the equation represents an ellipse -/
def ellipse_range (m : ℝ) : Prop :=
  (m > -1 ∧ m < 1/2) ∨ (m > 1/2 ∧ m < 2)

/-- Theorem stating that the equation represents an ellipse if and only if m is in the specified range -/
theorem ellipse_iff_range (m : ℝ) : is_ellipse m ↔ ellipse_range m := by
  sorry

end ellipse_iff_range_l3342_334272


namespace max_sum_of_square_roots_max_sum_of_square_roots_achievable_l3342_334281

theorem max_sum_of_square_roots (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 8) :
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ Real.sqrt 78 :=
by sorry

theorem max_sum_of_square_roots_achievable :
  ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 8 ∧
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = Real.sqrt 78 :=
by sorry

end max_sum_of_square_roots_max_sum_of_square_roots_achievable_l3342_334281
