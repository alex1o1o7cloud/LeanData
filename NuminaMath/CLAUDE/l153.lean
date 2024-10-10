import Mathlib

namespace horse_purchase_problem_l153_15346

/-- The problem of three people buying a horse -/
theorem horse_purchase_problem (x y z : ℚ) : 
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  (x = 60/17 ∧ y = 136/17 ∧ z = 156/17) := by
  sorry

end horse_purchase_problem_l153_15346


namespace unique_solution_l153_15374

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x + y - 1 = 0 ∧ x - 2*y + 2 = 0

/-- The solution to the system of equations -/
def solution : ℝ × ℝ := (0, 1)

/-- Theorem stating that the solution is unique and satisfies the system -/
theorem unique_solution :
  system solution.1 solution.2 ∧
  ∀ x y : ℝ, system x y → (x, y) = solution := by
  sorry

end unique_solution_l153_15374


namespace inequality_solution_set_l153_15310

theorem inequality_solution_set :
  {x : ℝ | 3 - 2*x > 7} = {x : ℝ | x < -2} := by
  sorry

end inequality_solution_set_l153_15310


namespace sum_of_solutions_l153_15358

noncomputable def solution_sum : ℝ → Prop :=
  fun x ↦ (x^2 - 6*x - 3 = 0) ∧ (x ≠ 1) ∧ (x ≠ -1)

theorem sum_of_solutions :
  ∃ (a b : ℝ), solution_sum a ∧ solution_sum b ∧ a + b = 6 :=
by sorry

end sum_of_solutions_l153_15358


namespace intersection_M_N_l153_15307

-- Define set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Define set N (domain of log|x|)
def N : Set ℝ := {x | x ≠ 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end intersection_M_N_l153_15307


namespace largest_prime_factor_of_pythagorean_triplet_number_l153_15326

/-- Given a three-digit number abc where a, b, and c are nonzero digits
    satisfying a^2 + b^2 = c^2, the largest possible prime factor of abc is 29. -/
theorem largest_prime_factor_of_pythagorean_triplet_number : ∃ (a b c : ℕ),
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (1 ≤ b ∧ b ≤ 9) ∧ 
  (1 ≤ c ∧ c ≤ 9) ∧ 
  a^2 + b^2 = c^2 ∧
  (∀ p : ℕ, p.Prime → p ∣ (100*a + 10*b + c) → p ≤ 29) ∧
  29 ∣ (100*a + 10*b + c) :=
sorry

end largest_prime_factor_of_pythagorean_triplet_number_l153_15326


namespace cupcake_price_is_two_l153_15329

/-- Calculates the price per cupcake given the number of trays, cupcakes per tray,
    fraction of cupcakes sold, and total earnings. -/
def price_per_cupcake (num_trays : ℕ) (cupcakes_per_tray : ℕ) 
                      (fraction_sold : ℚ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (fraction_sold * (num_trays * cupcakes_per_tray))

/-- Proves that the price per cupcake is $2 given the specific conditions. -/
theorem cupcake_price_is_two :
  price_per_cupcake 4 20 (3/5) 96 = 2 := by
  sorry

end cupcake_price_is_two_l153_15329


namespace dave_spent_on_mom_lunch_l153_15383

def derek_initial : ℕ := 40
def derek_lunch1 : ℕ := 14
def derek_dad_lunch : ℕ := 11
def derek_lunch2 : ℕ := 5
def dave_initial : ℕ := 50
def difference_left : ℕ := 33

theorem dave_spent_on_mom_lunch :
  dave_initial - (derek_initial - derek_lunch1 - derek_dad_lunch - derek_lunch2 + difference_left) = 7 := by
  sorry

end dave_spent_on_mom_lunch_l153_15383


namespace least_positive_integer_with_given_remainders_l153_15337

theorem least_positive_integer_with_given_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 4 = 1 ∧
  x % 5 = 2 ∧
  x % 6 = 3 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 1 ∧ y % 5 = 2 ∧ y % 6 = 3 → x ≤ y :=
by
  sorry

end least_positive_integer_with_given_remainders_l153_15337


namespace tan_theta_value_l153_15380

theorem tan_theta_value (θ : ℝ) (z : ℂ) : 
  z = Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5) → 
  z.re = 0 → 
  z.im ≠ 0 → 
  Real.tan θ = -3/4 :=
sorry

end tan_theta_value_l153_15380


namespace prob_all_even_before_odd_prob_all_even_before_odd_proof_l153_15311

/-- Represents an 8-sided die with numbers from 1 to 8 -/
inductive Die
| one | two | three | four | five | six | seven | eight

/-- Defines whether a number on the die is even or odd -/
def Die.isEven : Die → Bool
| Die.two => true
| Die.four => true
| Die.six => true
| Die.eight => true
| _ => false

/-- The probability of rolling an even number -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number -/
def probOdd : ℚ := 1/2

/-- The set of even numbers on the die -/
def evenNumbers : Set Die := {Die.two, Die.four, Die.six, Die.eight}

/-- Theorem: The probability of rolling every even number at least once
    before rolling any odd number on an 8-sided die is 1/384 -/
theorem prob_all_even_before_odd : ℚ :=
  1/384

/-- Proof of the theorem -/
theorem prob_all_even_before_odd_proof :
  prob_all_even_before_odd = 1/384 := by
  sorry

end prob_all_even_before_odd_prob_all_even_before_odd_proof_l153_15311


namespace smallest_quotient_by_18_l153_15372

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_quotient_by_18 (U : ℕ) (hU : is_binary_number U) (hDiv : U % 18 = 0) :
  ∃ Y : ℕ, Y = U / 18 ∧ Y ≥ 61728395 ∧ (∀ Z : ℕ, (∃ V : ℕ, is_binary_number V ∧ V % 18 = 0 ∧ Z = V / 18) → Z ≥ Y) :=
sorry

end smallest_quotient_by_18_l153_15372


namespace calculator_purchase_theorem_l153_15325

/-- Represents the unit price of a type A calculator -/
def price_A : ℝ := 110

/-- Represents the unit price of a type B calculator -/
def price_B : ℝ := 120

/-- Represents the total number of calculators to be purchased -/
def total_calculators : ℕ := 100

/-- Theorem stating the properties of calculator prices and minimum purchase cost -/
theorem calculator_purchase_theorem :
  (price_B = price_A + 10) ∧
  (550 / price_A = 600 / price_B) ∧
  (∀ a b : ℕ, a + b = total_calculators → b ≤ 3 * a →
    price_A * a + price_B * b ≥ 11000) :=
by sorry

end calculator_purchase_theorem_l153_15325


namespace g_inverse_sum_l153_15315

/-- The function g(x) defined piecewise -/
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 4 * x

/-- Theorem stating that c + d = 7.25 given the conditions -/
theorem g_inverse_sum (c d : ℝ) :
  (∀ x, g c d (g c d x) = x) →
  c + d = 7.25 := by
  sorry

end g_inverse_sum_l153_15315


namespace muffins_baked_by_macadams_class_l153_15332

theorem muffins_baked_by_macadams_class (brier_muffins flannery_muffins total_muffins : ℕ) 
  (h1 : brier_muffins = 18)
  (h2 : flannery_muffins = 17)
  (h3 : total_muffins = 55) :
  total_muffins - (brier_muffins + flannery_muffins) = 20 := by
  sorry

end muffins_baked_by_macadams_class_l153_15332


namespace q_transformation_l153_15390

theorem q_transformation (w d z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
  (h1 : ∀ w d z, q w d z = 5 * w / (4 * d * z^2))
  (h2 : ∃ k, q (k * w) (2 * d) (3 * z) = 0.2222222222222222 * q w d z) :
  ∃ k, k = 4 ∧ q (k * w) (2 * d) (3 * z) = 0.2222222222222222 * q w d z :=
sorry

end q_transformation_l153_15390


namespace ramon_age_in_twenty_years_ramon_current_age_l153_15324

/-- Ramon's current age -/
def ramon_age : ℕ := 26

/-- Loui's current age -/
def loui_age : ℕ := 23

/-- In twenty years, Ramon will be twice as old as Loui is today -/
theorem ramon_age_in_twenty_years (ramon_age loui_age : ℕ) :
  ramon_age + 20 = 2 * loui_age := by sorry

theorem ramon_current_age : ramon_age = 26 := by sorry

end ramon_age_in_twenty_years_ramon_current_age_l153_15324


namespace inequality_proof_l153_15344

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  a * Real.sqrt b + b * Real.sqrt c + c * Real.sqrt a ≤ 1 / Real.sqrt 3 := by
  sorry

end inequality_proof_l153_15344


namespace johns_number_l153_15336

theorem johns_number : ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 200 ∣ n ∧ 45 ∣ n ∧ n = 1800 := by
  sorry

end johns_number_l153_15336


namespace parabola_values_l153_15359

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_values (p : Parabola) : 
  (p.y_coord 4 = 5) ∧ 
  (p.y_coord 2 = -3) ∧ 
  (p.y_coord 6 = 3) ∧
  (∀ x : ℝ, p.y_coord x = p.y_coord (8 - x)) →
  p.a = -2 ∧ p.b = 16 ∧ p.c = -27 := by
  sorry

end parabola_values_l153_15359


namespace antonio_meatballs_l153_15320

/-- Given a recipe for meatballs and family size, calculate how many meatballs Antonio will eat -/
theorem antonio_meatballs (hamburger_per_meatball : ℚ) (family_size : ℕ) (total_hamburger : ℕ) :
  hamburger_per_meatball = 1/8 →
  family_size = 8 →
  total_hamburger = 4 →
  (total_hamburger / hamburger_per_meatball) / family_size = 4 :=
by sorry

end antonio_meatballs_l153_15320


namespace inequality_properties_l153_15356

theorem inequality_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) := by
  sorry

end inequality_properties_l153_15356


namespace quadratic_equation_solution_shift_l153_15349

theorem quadratic_equation_solution_shift 
  (m h k : ℝ) 
  (hm : m ≠ 0) 
  (h1 : m * (2 - h)^2 - k = 0) 
  (h2 : m * (5 - h)^2 - k = 0) :
  m * (1 - h + 1)^2 = k ∧ m * (4 - h + 1)^2 = k := by
sorry

end quadratic_equation_solution_shift_l153_15349


namespace factorial_500_trailing_zeros_l153_15303

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeros :
  trailingZeros 500 = 124 := by
  sorry

end factorial_500_trailing_zeros_l153_15303


namespace inscribed_cube_volume_is_sqrt_6_over_36_l153_15378

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (lateral_face_equilateral : Bool)

/-- A cube inscribed in a pyramid -/
structure InscribedCube :=
  (pyramid : Pyramid)
  (bottom_face_on_base : Bool)
  (top_face_edges_on_lateral_faces : Bool)

/-- The volume of an inscribed cube in a specific pyramid -/
noncomputable def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume_is_sqrt_6_over_36 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side = 1) 
  (h2 : cube.pyramid.lateral_face_equilateral = true)
  (h3 : cube.bottom_face_on_base = true)
  (h4 : cube.top_face_edges_on_lateral_faces = true) : 
  inscribed_cube_volume cube = Real.sqrt 6 / 36 :=
sorry

end inscribed_cube_volume_is_sqrt_6_over_36_l153_15378


namespace lillian_cupcakes_l153_15398

/-- Represents the number of dozen cupcakes Lillian can bake and ice --/
def cupcakes_dozen : ℕ := by sorry

theorem lillian_cupcakes :
  let initial_sugar : ℕ := 3
  let bags_bought : ℕ := 2
  let sugar_per_bag : ℕ := 6
  let sugar_for_batter : ℕ := 1
  let sugar_for_frosting : ℕ := 2
  
  let total_sugar : ℕ := initial_sugar + bags_bought * sugar_per_bag
  let sugar_per_dozen : ℕ := sugar_for_batter + sugar_for_frosting
  
  cupcakes_dozen = total_sugar / sugar_per_dozen ∧ cupcakes_dozen = 5 := by sorry

end lillian_cupcakes_l153_15398


namespace copper_ion_test_l153_15328

theorem copper_ion_test (total_beakers : ℕ) (copper_beakers : ℕ) (total_drops : ℕ) (non_copper_tested : ℕ) :
  total_beakers = 22 →
  copper_beakers = 8 →
  total_drops = 45 →
  non_copper_tested = 7 →
  (copper_beakers + non_copper_tested) * 3 = total_drops :=
by sorry

end copper_ion_test_l153_15328


namespace marcy_cat_time_l153_15331

/-- Given that Marcy spends 12 minutes petting her cat and 1/3 of that time combing it,
    prove that she spends 16 minutes in total with her cat. -/
theorem marcy_cat_time (petting_time : ℝ) (combing_ratio : ℝ) : 
  petting_time = 12 → combing_ratio = 1/3 → petting_time + combing_ratio * petting_time = 16 := by
sorry

end marcy_cat_time_l153_15331


namespace salon_cost_calculation_l153_15321

def salon_total_cost (manicure_cost pedicure_cost hair_treatment_cost : ℝ)
                     (manicure_tax_rate pedicure_tax_rate hair_treatment_tax_rate : ℝ)
                     (manicure_tip_rate pedicure_tip_rate hair_treatment_tip_rate : ℝ) : ℝ :=
  let manicure_total := manicure_cost * (1 + manicure_tax_rate + manicure_tip_rate)
  let pedicure_total := pedicure_cost * (1 + pedicure_tax_rate + pedicure_tip_rate)
  let hair_treatment_total := hair_treatment_cost * (1 + hair_treatment_tax_rate + hair_treatment_tip_rate)
  manicure_total + pedicure_total + hair_treatment_total

theorem salon_cost_calculation :
  salon_total_cost 30 40 50 0.05 0.07 0.09 0.25 0.20 0.15 = 151.80 := by
  sorry

end salon_cost_calculation_l153_15321


namespace whitewashing_cost_is_6342_l153_15341

/-- Calculates the cost of white washing a room with given dimensions, door, windows, and cost per square foot. -/
def whitewashing_cost (room_length room_width room_height : ℝ)
                      (door_width door_height : ℝ)
                      (window_width window_height : ℝ)
                      (num_windows : ℕ)
                      (cost_per_sqft : ℝ) : ℝ :=
  let total_wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_width * door_height
  let window_area := num_windows * (window_width * window_height)
  let net_area := total_wall_area - door_area - window_area
  net_area * cost_per_sqft

/-- Theorem stating that the cost of white washing the given room is 6342 Rs. -/
theorem whitewashing_cost_is_6342 :
  whitewashing_cost 25 15 12 6 3 4 3 3 7 = 6342 := by
  sorry

end whitewashing_cost_is_6342_l153_15341


namespace sum_positive_implies_at_least_one_positive_l153_15361

theorem sum_positive_implies_at_least_one_positive (x y : ℝ) : x + y > 0 → x > 0 ∨ y > 0 := by
  sorry

end sum_positive_implies_at_least_one_positive_l153_15361


namespace backyard_area_l153_15333

/-- A rectangular backyard satisfying certain conditions -/
structure Backyard where
  length : ℝ
  width : ℝ
  length_condition : 25 * length = 1000
  perimeter_condition : 10 * (2 * (length + width)) = 1000

/-- The area of a backyard is 400 square meters -/
theorem backyard_area (b : Backyard) : b.length * b.width = 400 := by
  sorry


end backyard_area_l153_15333


namespace smallest_four_digit_divisible_by_35_proof_l153_15396

/-- The smallest four-digit number divisible by 35 -/
def smallest_four_digit_divisible_by_35 : Nat := 1170

/-- A number is four digits if it's between 1000 and 9999 -/
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_35_proof :
  (is_four_digit smallest_four_digit_divisible_by_35) ∧ 
  (smallest_four_digit_divisible_by_35 % 35 = 0) ∧
  (∀ n : Nat, is_four_digit n → n % 35 = 0 → n ≥ smallest_four_digit_divisible_by_35) := by
  sorry

#eval smallest_four_digit_divisible_by_35

end smallest_four_digit_divisible_by_35_proof_l153_15396


namespace equation_solution_l153_15364

theorem equation_solution : ∃ x : ℝ, (((1 + x) / (2 - x)) - 1 = 1 / (x - 2)) ∧ x = 0 := by
  sorry

end equation_solution_l153_15364


namespace factorization_1_factorization_2_l153_15327

-- Part 1
theorem factorization_1 (a : ℝ) : (a^2 - 4*a + 4) - 4*(a - 2) + 4 = (a - 4)^2 := by
  sorry

-- Part 2
theorem factorization_2 (x y : ℝ) : 16*x^4 - 81*y^4 = (4*x^2 + 9*y^2)*(2*x + 3*y)*(2*x - 3*y) := by
  sorry

end factorization_1_factorization_2_l153_15327


namespace equation_solutions_l153_15308

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 5) * (x - 3) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 5) * (x - 3)
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → 
    (f x / g x = 1 ↔ x = 4 ∨ x = 4 + 2 * Real.sqrt 10 ∨ x = 4 - 2 * Real.sqrt 10) :=
by sorry

end equation_solutions_l153_15308


namespace min_value_x_plus_y_l153_15376

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 3/(y+2) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 3/(y₀+2) = 1 ∧ x₀ + y₀ = 2 + 2 * Real.sqrt 3 := by
  sorry

end min_value_x_plus_y_l153_15376


namespace repeating_decimal_to_fraction_l153_15353

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = (10^n * 6 - 6) / (10^n - 1)) → x = 2/3 := by
  sorry

end repeating_decimal_to_fraction_l153_15353


namespace tangent_segment_region_area_l153_15375

theorem tangent_segment_region_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 6) : 
  let outer_radius := r * Real.sqrt 2
  let area := π * (outer_radius^2 - r^2)
  area = 9 * π := by sorry

end tangent_segment_region_area_l153_15375


namespace twin_running_problem_l153_15313

theorem twin_running_problem (x : ℝ) :
  (x ≥ 0) →  -- Ensure distance is non-negative
  (2 * x = 25) →  -- Final distance equation
  (x = 12.5) :=
by
  sorry

end twin_running_problem_l153_15313


namespace probability_theorem_l153_15309

def family_A_size : ℕ := 5
def family_B_size : ℕ := 3
def total_girls : ℕ := 5
def total_boys : ℕ := 3

def probability_at_least_one_family_all_girls : ℚ :=
  11 / 56

theorem probability_theorem :
  let total_children := family_A_size + family_B_size
  probability_at_least_one_family_all_girls = 11 / 56 :=
by sorry

end probability_theorem_l153_15309


namespace puppies_adopted_per_day_l153_15335

theorem puppies_adopted_per_day 
  (initial_puppies : ℕ) 
  (additional_puppies : ℕ) 
  (adoption_days : ℕ) 
  (h1 : initial_puppies = 2)
  (h2 : additional_puppies = 34)
  (h3 : adoption_days = 9)
  (h4 : (initial_puppies + additional_puppies) % adoption_days = 0) :
  (initial_puppies + additional_puppies) / adoption_days = 4 := by
sorry

end puppies_adopted_per_day_l153_15335


namespace tangent_line_at_e_l153_15314

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_e :
  let p : ℝ × ℝ := (Real.exp 1, f (Real.exp 1))
  let m : ℝ := deriv f (Real.exp 1)
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  tangent_line = λ x => 2 * x - Real.exp 1 :=
sorry

end tangent_line_at_e_l153_15314


namespace sum_inequality_l153_15334

theorem sum_inequality (a b c : ℝ) (k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a * b * c = 1) (hk : k ≥ 3) : 
  (1 / (a^k * (b + c)) + 1 / (b^k * (a + c)) + 1 / (c^k * (a + b))) ≥ 3/2 := by
  sorry

end sum_inequality_l153_15334


namespace sum_of_roots_equation_l153_15322

theorem sum_of_roots_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7)
  (∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) →
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = -2) :=
by sorry

end sum_of_roots_equation_l153_15322


namespace betty_age_l153_15387

/-- Given the ages of Alice, Betty, and Carol satisfying certain conditions,
    prove that Betty's age is 7.5 years. -/
theorem betty_age (alice carol betty : ℝ) 
    (h1 : carol = 5 * alice)
    (h2 : carol = 2 * betty)
    (h3 : alice = carol - 12) : 
  betty = 7.5 := by
  sorry

end betty_age_l153_15387


namespace oranges_picked_total_l153_15354

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 14

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 55 := by sorry

end oranges_picked_total_l153_15354


namespace f_f_3_equals_13_9_l153_15377

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

theorem f_f_3_equals_13_9 : f (f 3) = 13/9 := by
  sorry

end f_f_3_equals_13_9_l153_15377


namespace max_intersections_six_paths_l153_15312

/-- The number of intersection points for a given number of paths -/
def intersection_points (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: With 6 paths, where each path intersects with every other path
    exactly once, the maximum number of intersection points is 15 -/
theorem max_intersections_six_paths :
  intersection_points 6 = 15 := by
  sorry

#eval intersection_points 6  -- This will output 15

end max_intersections_six_paths_l153_15312


namespace statements_are_false_l153_15347

theorem statements_are_false : 
  (¬ ∀ (x : ℚ), ∃ (y : ℚ), (x < y ∧ y < -x) ∨ (-x < y ∧ y < x)) ∧ 
  (¬ ∀ (x : ℚ), x ≠ 0 → ∃ (y : ℚ), (x < y ∧ y < x⁻¹) ∨ (x⁻¹ < y ∧ y < x)) :=
by sorry

end statements_are_false_l153_15347


namespace matthews_cakes_equal_crackers_l153_15379

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 4

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 32

/-- The number of crackers each person ate -/
def crackers_per_person : ℕ := 8

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := initial_crackers

theorem matthews_cakes_equal_crackers :
  initial_cakes = initial_crackers :=
by sorry

end matthews_cakes_equal_crackers_l153_15379


namespace senior_mean_score_senior_mean_score_is_88_l153_15381

/-- The mean score of seniors in a math competition --/
theorem senior_mean_score (total_students : ℕ) (overall_mean : ℝ) 
  (junior_ratio : ℝ) (senior_score_ratio : ℝ) : ℝ :=
  let senior_count := (total_students : ℝ) / (1 + junior_ratio)
  let junior_count := senior_count * junior_ratio
  let junior_mean := overall_mean * (total_students : ℝ) / (senior_count * senior_score_ratio + junior_count)
  junior_mean * senior_score_ratio

/-- The mean score of seniors is approximately 88 --/
theorem senior_mean_score_is_88 : 
  ∃ ε > 0, |senior_mean_score 150 80 1.2 1.2 - 88| < ε :=
by
  sorry

end senior_mean_score_senior_mean_score_is_88_l153_15381


namespace range_of_x_l153_15304

def P (x : ℝ) : Prop := (x + 1) / (x - 3) ≥ 0

def Q (x : ℝ) : Prop := |1 - x/2| < 1

theorem range_of_x (x : ℝ) : 
  P x ∧ ¬Q x ↔ x ≤ -1 ∨ x ≥ 4 :=
by sorry

end range_of_x_l153_15304


namespace village_population_equality_second_village_initial_population_l153_15399

/-- The initial population of Village X -/
def initial_pop_X : ℕ := 68000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The yearly increase in population of the second village -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 13

/-- The initial population of the second village -/
def initial_pop_Y : ℕ := 42000

theorem village_population_equality :
  initial_pop_X - years * decrease_rate_X = initial_pop_Y + years * increase_rate_Y :=
by sorry

theorem second_village_initial_population :
  initial_pop_Y = 42000 :=
by sorry

end village_population_equality_second_village_initial_population_l153_15399


namespace first_day_is_saturday_l153_15382

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  saturdays : Nat
  sundays : Nat

/-- Theorem: In a 30-day month with 5 Saturdays and 5 Sundays, the first day is Saturday -/
theorem first_day_is_saturday (m : Month) (h1 : m.days = 30) (h2 : m.saturdays = 5) (h3 : m.sundays = 5) :
  ∃ (first_day : DayOfWeek), first_day = DayOfWeek.Saturday := by
  sorry


end first_day_is_saturday_l153_15382


namespace anna_coins_value_l153_15395

/-- Represents the number and value of coins Anna has. -/
structure Coins where
  pennies : ℕ
  nickels : ℕ
  total : ℕ
  penny_nickel_relation : pennies = 2 * (nickels + 1) + 1
  total_coins : pennies + nickels = total

/-- The value of Anna's coins in cents -/
def coin_value (c : Coins) : ℕ := c.pennies + 5 * c.nickels

/-- Theorem stating that Anna's coins are worth 31 cents -/
theorem anna_coins_value :
  ∃ c : Coins, c.total = 15 ∧ coin_value c = 31 := by
  sorry


end anna_coins_value_l153_15395


namespace multiplicative_inverse_of_3_mod_47_l153_15363

theorem multiplicative_inverse_of_3_mod_47 : ∃ x : ℕ, x < 47 ∧ (3 * x) % 47 = 1 :=
by
  use 16
  sorry

end multiplicative_inverse_of_3_mod_47_l153_15363


namespace system_no_solution_l153_15348

def has_no_solution (a b c : ℤ) : Prop :=
  2 / a = -b / 5 ∧ -b / 5 = 1 / -c ∧ 2 / a ≠ 2 * b / a

theorem system_no_solution : 
  {(a, b, c) : ℤ × ℤ × ℤ | has_no_solution a b c} = 
  {(-2, 5, 1), (2, -5, -1), (10, -1, -5)} := by sorry

end system_no_solution_l153_15348


namespace factorization_equality_l153_15306

theorem factorization_equality (p : ℝ) : (p - 4) * (p + 1) + 3 * p = (p + 2) * (p - 2) := by
  sorry

end factorization_equality_l153_15306


namespace angle_A_is_pi_third_max_area_is_sqrt_three_max_area_achieved_l153_15385

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 2 ∧ (2 + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C

-- Theorem 1: Angle A is π/3
theorem angle_A_is_pi_third (t : Triangle) (h : satisfiesConditions t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Maximum area is √3
theorem max_area_is_sqrt_three (t : Triangle) (h : satisfiesConditions t) : 
  (1/2 * t.b * t.c * Real.sin t.A) ≤ Real.sqrt 3 := by
  sorry

-- Theorem 2 (continued): The maximum area is achieved
theorem max_area_achieved (t : Triangle) : 
  ∃ (t : Triangle), satisfiesConditions t ∧ (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 := by
  sorry

end angle_A_is_pi_third_max_area_is_sqrt_three_max_area_achieved_l153_15385


namespace fib_sum_eq_49_287_l153_15339

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of G_n / 7^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 7^n

theorem fib_sum_eq_49_287 : fibSum = 49 / 287 := by sorry

end fib_sum_eq_49_287_l153_15339


namespace zero_of_f_floor_l153_15362

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem zero_of_f_floor (x : ℝ) (hx : f x = 0) : Int.floor x = 2 := by
  sorry

end zero_of_f_floor_l153_15362


namespace solve_for_a_l153_15323

theorem solve_for_a (x a : ℝ) (h1 : 2 * x - 5 * a = 3 * a + 22) (h2 : x = 3) : a = -2 := by
  sorry

end solve_for_a_l153_15323


namespace equation_solutions_l153_15397

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)
  let g (x : ℝ) := (x - 2) * (x - 4) * (x - 2) * (x - 5)
  ∀ x : ℝ, (g x ≠ 0 ∧ f x / g x = 1) ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end equation_solutions_l153_15397


namespace restaurant_order_combinations_l153_15393

def menu_size : ℕ := 15
def num_people : ℕ := 3

theorem restaurant_order_combinations :
  menu_size ^ num_people = 3375 := by sorry

end restaurant_order_combinations_l153_15393


namespace complex_fraction_simplification_l153_15319

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (7 + 18 * i) / (3 - 4 * i) = -51/25 + 82/25 * i := by
  sorry

end complex_fraction_simplification_l153_15319


namespace x0_value_l153_15373

def f (x : ℝ) : ℝ := x^3 + x - 1

theorem x0_value (x₀ : ℝ) (h : (deriv f) x₀ = 4) : x₀ = 1 ∨ x₀ = -1 := by
  sorry

end x0_value_l153_15373


namespace perpendicular_vectors_k_value_l153_15330

/-- Given two 2D vectors a and b, where a = (2,1), a + b = (1,k), and a ⟂ b, prove that k = 3 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  a + b = (1, k) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  k = 3 := by sorry

end perpendicular_vectors_k_value_l153_15330


namespace student_arrangement_count_l153_15369

/-- The number of ways to select and arrange students with non-adjacent boys -/
def student_arrangements (num_boys num_girls select_boys select_girls : ℕ) : ℕ :=
  Nat.choose num_boys select_boys *
  Nat.choose num_girls select_girls *
  Nat.factorial select_girls *
  Nat.factorial (select_girls + 1)

/-- Theorem: The number of arrangements of 2 boys from 4 and 3 girls from 6,
    where the boys are not adjacent, is 8640 -/
theorem student_arrangement_count :
  student_arrangements 4 6 2 3 = 8640 := by
sorry

end student_arrangement_count_l153_15369


namespace hyperbola_focal_length_l153_15367

/-- The focal length of a hyperbola with equation y²/4 - x² = 1 is 2√5 -/
theorem hyperbola_focal_length :
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / 4 - x^2 = 1}
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ hyperbola → q ∈ hyperbola → 
      abs (dist p (0, f) - dist p (0, -f)) = 2 * abs (p.1) :=
by sorry

end hyperbola_focal_length_l153_15367


namespace problem_statement_l153_15302

theorem problem_statement (x : ℝ) :
  x = (Real.sqrt (6 + 2 * Real.sqrt 5) + Real.sqrt (6 - 2 * Real.sqrt 5)) / Real.sqrt 20 →
  (1 + x^5 - x^7)^(2012^(3^11)) = 1 := by
  sorry

end problem_statement_l153_15302


namespace emily_beads_count_l153_15316

theorem emily_beads_count (beads_per_necklace : ℕ) (necklaces_made : ℕ) (total_beads : ℕ) : 
  beads_per_necklace = 8 → necklaces_made = 2 → total_beads = beads_per_necklace * necklaces_made → total_beads = 16 := by
  sorry

end emily_beads_count_l153_15316


namespace total_miles_driven_l153_15368

/-- The total miles driven by Darius and Julia -/
def total_miles (darius_miles julia_miles : ℕ) : ℕ :=
  darius_miles + julia_miles

/-- Theorem stating that the total miles driven by Darius and Julia is 1677 -/
theorem total_miles_driven :
  total_miles 679 998 = 1677 := by
  sorry

end total_miles_driven_l153_15368


namespace triangle_base_length_l153_15342

theorem triangle_base_length (height : ℝ) (area : ℝ) : 
  height = 8 → area = 24 → (1/2) * 6 * height = area :=
by
  sorry

end triangle_base_length_l153_15342


namespace cross_shape_surface_area_l153_15305

/-- Represents a 3D shape made of unit cubes -/
structure CubeShape where
  num_cubes : ℕ
  exposed_faces : ℕ

/-- The cross-like shape made of 5 unit cubes -/
def cross_shape : CubeShape :=
  { num_cubes := 5,
    exposed_faces := 22 }

/-- Theorem stating that the surface area of the cross-shape is 22 square units -/
theorem cross_shape_surface_area :
  cross_shape.exposed_faces = 22 := by
  sorry

end cross_shape_surface_area_l153_15305


namespace shirt_sale_price_l153_15317

theorem shirt_sale_price (original_price : ℝ) (initial_sale_price : ℝ) 
  (h1 : initial_sale_price > 0)
  (h2 : original_price > 0)
  (h3 : initial_sale_price * 0.8 = original_price * 0.64) :
  initial_sale_price / original_price = 0.8 := by
sorry

end shirt_sale_price_l153_15317


namespace conditioner_shampoo_ratio_l153_15360

/-- Proves the ratio of daily conditioner use to daily shampoo use -/
theorem conditioner_shampoo_ratio 
  (daily_shampoo : ℝ) 
  (total_volume : ℝ) 
  (days : ℕ) 
  (h1 : daily_shampoo = 1)
  (h2 : total_volume = 21)
  (h3 : days = 14) :
  (total_volume - daily_shampoo * days) / days / daily_shampoo = 1 / 2 := by
sorry

end conditioner_shampoo_ratio_l153_15360


namespace line_proof_l153_15338

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := 3*x + y - 2 = 0

-- Define the line to be proved
def prove_line (x y : ℝ) : Prop := x - 3*y + 1 = 0

-- Function to get the center of a circle
def circle_center (circle : (ℝ → ℝ → Prop)) : ℝ × ℝ := sorry

-- Function to check if two lines are perpendicular
def perpendicular (line1 line2 : ℝ → ℝ → Prop) : Prop := sorry

theorem line_proof :
  let center := circle_center circle_C
  prove_line center.1 center.2 ∧ 
  perpendicular prove_line given_line := by sorry

end line_proof_l153_15338


namespace circle_center_l153_15318

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 5

/-- Theorem: The center of the circle defined by x^2 + y^2 - 2x - 4y = 0 is at (1, 2) -/
theorem circle_center : CircleCenter 1 2 := by
  sorry

end circle_center_l153_15318


namespace initial_work_plan_l153_15365

/-- Proves that the initial plan was to complete the work in 28 days given the conditions of the problem. -/
theorem initial_work_plan (total_men : Nat) (absent_men : Nat) (days_with_reduced_men : Nat) 
  (h1 : total_men = 42)
  (h2 : absent_men = 6)
  (h3 : days_with_reduced_men = 14) : 
  (total_men * ((total_men - absent_men) * days_with_reduced_men)) / (total_men - absent_men) = 28 := by
  sorry

#eval (42 * ((42 - 6) * 14)) / (42 - 6)

end initial_work_plan_l153_15365


namespace complement_of_M_l153_15343

-- Define the set M
def M : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end complement_of_M_l153_15343


namespace centroid_incenter_ratio_l153_15388

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumcenter, incenter, and centroid
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def incenter (t : Triangle) : ℝ × ℝ := sorry
def centroid_of_arc_midpoints (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem centroid_incenter_ratio (t : Triangle) :
  let O := circumcenter t
  let I := incenter t
  let G := centroid_of_arc_midpoints t
  distance A B = 13 →
  distance B C = 14 →
  distance C A = 15 →
  (distance G O) / (distance G I) = 1 / 4 := by
  sorry

end centroid_incenter_ratio_l153_15388


namespace visit_probability_l153_15350

/-- The probability of Jen visiting either Chile or Madagascar, but not both -/
theorem visit_probability (p_chile p_madagascar : ℝ) 
  (h_chile : p_chile = 0.30)
  (h_madagascar : p_madagascar = 0.50) : 
  p_chile + p_madagascar - p_chile * p_madagascar = 0.65 := by
  sorry

end visit_probability_l153_15350


namespace triangle_theorem_l153_15389

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.a = 2 * Real.sqrt 3) 
  (h3 : t.b = 2) : 
  t.A = Real.pi / 3 ∧ Real.cos t.C = 0 := by
  sorry


end triangle_theorem_l153_15389


namespace complement_intersection_theorem_l153_15394

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3}
def N : Set ℕ := {1, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end complement_intersection_theorem_l153_15394


namespace marker_problem_l153_15384

theorem marker_problem :
  ∃ (n : ℕ) (p : ℝ), 
    p > 0 ∧
    3.51 = p * n ∧
    4.25 = p * (n + 4) ∧
    n > 0 := by
  sorry

end marker_problem_l153_15384


namespace remainder_sum_l153_15370

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 74) 
  (hd : d % 120 = 114) : 
  (c + d) % 40 = 28 := by
sorry

end remainder_sum_l153_15370


namespace necessary_but_not_sufficient_l153_15345

theorem necessary_but_not_sufficient 
  (a b : ℝ) : 
  (((b + 2) / (a + 2) > b / a) ↔ (a > b ∧ b > 0)) → False :=
by sorry

end necessary_but_not_sufficient_l153_15345


namespace negative_a_sixth_div_a_cube_l153_15300

theorem negative_a_sixth_div_a_cube (a : ℝ) : (-a)^6 / a^3 = a^3 := by sorry

end negative_a_sixth_div_a_cube_l153_15300


namespace intersection_in_sphere_l153_15355

/-- Given three unit cylinders with pairwise perpendicular axes, 
    their intersection is contained in a sphere of radius √(3/2) --/
theorem intersection_in_sphere (a b c d e f : ℝ) :
  ∀ x y z : ℝ, 
  (x - a)^2 + (y - b)^2 ≤ 1 →
  (y - c)^2 + (z - d)^2 ≤ 1 →
  (z - e)^2 + (x - f)^2 ≤ 1 →
  ∃ center_x center_y center_z : ℝ, 
    (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 ≤ 3/2 := by
  sorry

end intersection_in_sphere_l153_15355


namespace arithmetic_geometric_inequality_l153_15340

theorem arithmetic_geometric_inequality (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (arith_prog : ∃ r : ℝ, b = a + r ∧ c = a + 2*r ∧ d = a + 3*r)
  (geom_prog : ∃ q : ℝ, e = a * q ∧ f = a * q^2 ∧ d = a * q^3) :
  b * c ≥ e * f := by
  sorry

end arithmetic_geometric_inequality_l153_15340


namespace cubic_root_problem_l153_15391

theorem cubic_root_problem (a b r s : ℤ) : 
  a ≠ 0 → b ≠ 0 → 
  (∀ x : ℤ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) →
  (r = s ∨ r = -2 ∨ s = -2) →
  (|a*b| = 272) :=
sorry

end cubic_root_problem_l153_15391


namespace sue_necklace_beads_l153_15301

def necklace_beads (purple blue green red : ℕ) : Prop :=
  (blue = 2 * purple) ∧
  (green = blue + 11) ∧
  (red = green / 2) ∧
  (purple + blue + green + red = 58)

theorem sue_necklace_beads :
  ∃ (purple blue green red : ℕ),
    purple = 7 ∧
    necklace_beads purple blue green red :=
by sorry

end sue_necklace_beads_l153_15301


namespace inequality_proof_l153_15392

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1/2) :
  (Real.sqrt x) / (4 * x + 1) + (Real.sqrt y) / (4 * y + 1) + (Real.sqrt z) / (4 * z + 1) ≤ 3 * Real.sqrt 6 / 10 :=
by sorry

end inequality_proof_l153_15392


namespace pythagorean_triple_even_l153_15366

theorem pythagorean_triple_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : Even x ∨ Even y := by
  sorry

end pythagorean_triple_even_l153_15366


namespace fraction_problem_l153_15371

theorem fraction_problem : ∃ x : ℚ, (0.60 * 40 : ℚ) = x * 25 + 4 :=
  sorry

end fraction_problem_l153_15371


namespace min_distance_sum_l153_15352

theorem min_distance_sum (x : ℝ) : 
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_distance_sum_l153_15352


namespace line_intersections_l153_15357

theorem line_intersections : 
  let line1 : ℝ → ℝ := λ x => 5 * x - 20
  let line2 : ℝ → ℝ := λ x => 190 - 3 * x
  let line3 : ℝ → ℝ := λ x => 2 * x + 15
  ∃ (x1 x2 : ℝ), 
    (line1 x1 = line2 x1 ∧ x1 = 105 / 4) ∧
    (line1 x2 = line3 x2 ∧ x2 = 35 / 3) := by
  sorry

end line_intersections_l153_15357


namespace largest_y_value_l153_15386

theorem largest_y_value (y : ℝ) : 
  5 * (4 * y^2 + 12 * y + 15) = y * (4 * y - 25) →
  y ≤ (-85 + 5 * Real.sqrt 97) / 32 :=
by sorry

end largest_y_value_l153_15386


namespace min_perimeter_of_triangle_l153_15351

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- A point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- The center of the ellipse -/
def center : ℝ × ℝ := (0, 0)

/-- A line passing through the center of the ellipse -/
structure LineThroughCenter where
  slope : ℝ

/-- Intersection points of the line with the ellipse -/
def intersectionPoints (l : LineThroughCenter) : PointOnEllipse × PointOnEllipse := sorry

/-- One of the foci of the ellipse -/
def focus : ℝ × ℝ := (3, 0)

/-- The perimeter of the triangle formed by two points on the ellipse and the focus -/
def trianglePerimeter (p q : PointOnEllipse) : ℝ := sorry

/-- The statement to be proved -/
theorem min_perimeter_of_triangle : 
  ∀ l : LineThroughCenter, 
  let (p, q) := intersectionPoints l
  18 ≤ trianglePerimeter p q ∧ 
  ∃ l₀ : LineThroughCenter, trianglePerimeter (intersectionPoints l₀).1 (intersectionPoints l₀).2 = 18 :=
sorry

end min_perimeter_of_triangle_l153_15351
