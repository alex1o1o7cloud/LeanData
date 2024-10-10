import Mathlib

namespace train_crossing_time_l93_9311

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 56 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end train_crossing_time_l93_9311


namespace sum_of_nine_terms_l93_9399

/-- An arithmetic sequence with sum Sₙ of first n terms, where a₄ = 9 and a₆ = 11 -/
structure ArithSeq where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)
  a4_eq_9 : a 4 = 9
  a6_eq_11 : a 6 = 11

/-- The sum of the first 9 terms of the arithmetic sequence is 90 -/
theorem sum_of_nine_terms (seq : ArithSeq) : seq.S 9 = 90 := by
  sorry

end sum_of_nine_terms_l93_9399


namespace no_solutions_exist_l93_9350

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 2) := by
  sorry

end no_solutions_exist_l93_9350


namespace ellipse_theorem_l93_9338

/-- Ellipse with focus at (-√3, 0) and point (1, y) on it --/
structure Ellipse where
  a : ℝ
  b : ℝ
  y : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_y_pos : y > 0
  h_eq : 1 / a^2 + y^2 / b^2 = 1
  h_focus : -Real.sqrt 3 = -Real.sqrt (a^2 - b^2)
  h_area : 1/2 * Real.sqrt 3 * y = 3/4

/-- The main theorem --/
theorem ellipse_theorem (e : Ellipse) :
  (∃ (k : ℝ), (∀ (x y : ℝ),
    (x^2 / 4 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
    (y = k * (x - 2) →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁^2 / 4 + y₁^2 = 1 ∧
        x₂^2 / 4 + y₂^2 = 1 ∧
        y₁ = k * (x₁ - 2) ∧
        y₂ = k * (x₂ - 2) ∧
        ∃ (t₁ t₂ : ℝ),
          t₁^2 + t₂^2 = 1 ∧
          t₁ = Real.sqrt 5 / 5 * (2 + x₂) ∧
          t₂ = Real.sqrt 5 / 5 * (y₂)))) ∧
  (k = 1/2 ∨ k = -1/2) := by sorry

end ellipse_theorem_l93_9338


namespace john_car_profit_l93_9376

/-- Calculates the profit from fixing and racing a car given the following parameters:
    * original_cost: The original cost to fix the car
    * discount_percentage: The discount percentage on the repair cost
    * prize_money: The total prize money won
    * kept_percentage: The percentage of prize money kept by the racer
-/
def calculate_car_profit (original_cost discount_percentage prize_money kept_percentage : ℚ) : ℚ :=
  let discounted_cost := original_cost * (1 - discount_percentage / 100)
  let kept_prize := prize_money * (kept_percentage / 100)
  kept_prize - discounted_cost

/-- Theorem stating that given the specific conditions of John's car repair and race,
    his profit is $47,000 -/
theorem john_car_profit :
  calculate_car_profit 20000 20 70000 90 = 47000 := by
  sorry

end john_car_profit_l93_9376


namespace equation_transformation_l93_9375

theorem equation_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 4 * b) :
  a / 4 = b / 3 := by
  sorry

end equation_transformation_l93_9375


namespace swimming_practice_months_l93_9384

theorem swimming_practice_months (total_required : ℕ) (completed : ℕ) (monthly_practice : ℕ) : 
  total_required = 1500 →
  completed = 180 →
  monthly_practice = 220 →
  (total_required - completed) / monthly_practice = 6 := by
sorry

end swimming_practice_months_l93_9384


namespace num_pyramids_eq_106_l93_9309

/-- A rectangular solid (cuboid) -/
structure Cuboid where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 2 × Fin 8)
  faces : Finset (Fin 6)

/-- A pyramid formed by vertices of a cuboid -/
structure Pyramid where
  vertices : Finset (Fin 4)

/-- The set of all possible pyramids formed from a cuboid -/
def all_pyramids (c : Cuboid) : Finset Pyramid :=
  sorry

/-- The number of different pyramids that can be formed from a cuboid -/
def num_pyramids (c : Cuboid) : ℕ :=
  (all_pyramids c).card

/-- Theorem: The number of different pyramids that can be formed
    using the vertices of a rectangular solid is equal to 106 -/
theorem num_pyramids_eq_106 (c : Cuboid) : num_pyramids c = 106 := by
  sorry

end num_pyramids_eq_106_l93_9309


namespace women_per_table_l93_9303

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 6 →
  men_per_table = 5 →
  total_customers = 48 →
  ∃ (women_per_table : ℕ),
    women_per_table * num_tables + men_per_table * num_tables = total_customers ∧
    women_per_table = 3 :=
by
  sorry

end women_per_table_l93_9303


namespace power_inequality_l93_9313

theorem power_inequality (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end power_inequality_l93_9313


namespace zero_exponent_l93_9377

theorem zero_exponent (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end zero_exponent_l93_9377


namespace sum_of_reciprocals_l93_9302

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end sum_of_reciprocals_l93_9302


namespace mika_stickers_decoration_l93_9344

/-- The number of stickers Mika used to decorate the greeting card -/
def stickers_used_for_decoration (initial : ℕ) (bought : ℕ) (received : ℕ) (given_away : ℕ) (left : ℕ) : ℕ :=
  initial + bought + received - given_away - left

/-- Theorem stating that Mika used 58 stickers to decorate the greeting card -/
theorem mika_stickers_decoration :
  stickers_used_for_decoration 20 26 20 6 2 = 58 := by
  sorry

end mika_stickers_decoration_l93_9344


namespace f_symmetry_l93_9381

/-- Given a function f(x) = x^3 + 2x, prove that f(a) + f(-a) = 0 for any real number a -/
theorem f_symmetry (a : ℝ) : (fun x : ℝ ↦ x^3 + 2*x) a + (fun x : ℝ ↦ x^3 + 2*x) (-a) = 0 := by
  sorry

end f_symmetry_l93_9381


namespace orange_bin_count_l93_9326

theorem orange_bin_count (initial : ℕ) (removed : ℕ) (added : ℕ) :
  initial = 40 →
  removed = 37 →
  added = 7 →
  initial - removed + added = 10 := by
  sorry

end orange_bin_count_l93_9326


namespace mark_additional_spending_l93_9321

def mark_spending (initial_amount : ℝ) (additional_first_store : ℝ) : Prop :=
  let half_spent := initial_amount / 2
  let remaining_after_half := initial_amount - half_spent
  let remaining_after_first := remaining_after_half - additional_first_store
  let third_spent := initial_amount / 3
  let remaining_after_second := remaining_after_first - third_spent - 16
  remaining_after_second = 0

theorem mark_additional_spending :
  mark_spending 180 14 := by sorry

end mark_additional_spending_l93_9321


namespace equal_value_proof_l93_9323

theorem equal_value_proof (a b : ℝ) (h1 : 10 * a = 6 * b) (h2 : 120 * a * b = 800) :
  10 * a = 20 ∧ 6 * b = 20 := by
  sorry

end equal_value_proof_l93_9323


namespace even_decreasing_comparison_l93_9308

-- Define an even function that is decreasing on (-∞, 0)
def even_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x < y ∧ y ≤ 0 → f y < f x)

-- Theorem statement
theorem even_decreasing_comparison 
  (f : ℝ → ℝ) 
  (h : even_decreasing_function f) : 
  f 2 < f (-3) := by
sorry

end even_decreasing_comparison_l93_9308


namespace trigonometric_identity_l93_9367

theorem trigonometric_identity : 
  Real.sin (4/3 * π) * Real.cos (11/6 * π) * Real.tan (3/4 * π) = 3/4 := by
  sorry

end trigonometric_identity_l93_9367


namespace beef_weight_calculation_l93_9368

theorem beef_weight_calculation (weight_after : ℝ) (percent_lost : ℝ) 
  (h1 : weight_after = 640)
  (h2 : percent_lost = 20) : 
  weight_after / (1 - percent_lost / 100) = 800 := by
  sorry

end beef_weight_calculation_l93_9368


namespace expression_evaluation_l93_9349

theorem expression_evaluation (a b : ℤ) (ha : a = 4) (hb : b = -2) :
  -a - b^4 + a*b = -28 := by sorry

end expression_evaluation_l93_9349


namespace negation_equivalence_l93_9317

theorem negation_equivalence (x : ℝ) :
  ¬(x = 0 ∨ x = 1 → x^2 - x = 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) := by
  sorry

end negation_equivalence_l93_9317


namespace attendees_count_l93_9395

/-- The number of people attending the family reunion --/
def attendees : ℕ := sorry

/-- The number of cans in each box of soda --/
def cans_per_box : ℕ := 10

/-- The cost of each box of soda in dollars --/
def cost_per_box : ℕ := 2

/-- The number of cans each person consumes --/
def cans_per_person : ℕ := 2

/-- The number of family members paying for the soda --/
def paying_family_members : ℕ := 6

/-- The amount each family member pays in dollars --/
def payment_per_member : ℕ := 4

/-- Theorem stating that the number of attendees is 60 --/
theorem attendees_count : attendees = 60 := by sorry

end attendees_count_l93_9395


namespace matrix_power_101_l93_9307

open Matrix

/-- Given a 3x3 matrix A, prove that A^101 equals the given result -/
theorem matrix_power_101 (A : Matrix (Fin 3) (Fin 3) ℝ) :
  A = ![![0, 0, 1],
       ![1, 0, 0],
       ![0, 1, 0]] →
  A^101 = ![![0, 1, 0],
            ![0, 0, 1],
            ![1, 0, 0]] := by
  sorry

end matrix_power_101_l93_9307


namespace ac_equals_twelve_l93_9363

theorem ac_equals_twelve (a b c d : ℝ) 
  (h1 : a = 2 * b)
  (h2 : c = d * b)
  (h3 : d + d = b * c)
  (h4 : d = 3) : 
  a * c = 12 := by
sorry

end ac_equals_twelve_l93_9363


namespace least_possible_beta_l93_9327

-- Define a structure for the right triangle
structure RightTriangle where
  alpha : ℕ
  beta : ℕ
  is_right_triangle : alpha + beta = 100
  alpha_prime : Nat.Prime alpha
  beta_prime : Nat.Prime beta
  alpha_odd : Odd alpha
  beta_odd : Odd beta
  alpha_greater : alpha > beta

-- Define the theorem
theorem least_possible_beta (t : RightTriangle) : 
  ∃ (min_beta : ℕ), min_beta = 3 ∧ 
  ∀ (valid_triangle : RightTriangle), valid_triangle.beta ≥ min_beta :=
sorry

end least_possible_beta_l93_9327


namespace computer_distribution_l93_9396

def distribute_computers (n : ℕ) (k : ℕ) (min : ℕ) : ℕ :=
  -- The number of ways to distribute n identical items among k recipients,
  -- with each recipient receiving at least min items
  sorry

theorem computer_distribution :
  distribute_computers 9 3 2 = 10 := by sorry

end computer_distribution_l93_9396


namespace rational_square_difference_l93_9383

theorem rational_square_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by
  sorry

end rational_square_difference_l93_9383


namespace sum_of_sqrt_inequality_l93_9382

theorem sum_of_sqrt_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) < 5 := by
  sorry

end sum_of_sqrt_inequality_l93_9382


namespace fraction_power_five_l93_9348

theorem fraction_power_five : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end fraction_power_five_l93_9348


namespace kekai_mms_packs_l93_9332

/-- The number of sundaes made on Monday -/
def monday_sundaes : ℕ := 40

/-- The number of m&ms per sundae on Monday -/
def monday_mms_per_sundae : ℕ := 6

/-- The number of sundaes made on Tuesday -/
def tuesday_sundaes : ℕ := 20

/-- The number of m&ms per sundae on Tuesday -/
def tuesday_mms_per_sundae : ℕ := 10

/-- The number of m&ms in each pack -/
def mms_per_pack : ℕ := 40

/-- The total number of m&m packs used -/
def total_packs_used : ℕ := 11

theorem kekai_mms_packs :
  (monday_sundaes * monday_mms_per_sundae + tuesday_sundaes * tuesday_mms_per_sundae) / mms_per_pack = total_packs_used :=
by sorry

end kekai_mms_packs_l93_9332


namespace inequality_system_solution_l93_9371

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℤ, x > 2*m ∧ x ≥ m - 3 ∧ (∀ y : ℤ, y > 2*m ∧ y ≥ m - 3 → y ≥ x) ∧ x = 1) 
  ↔ 0 ≤ m ∧ m < 1/2 := by
sorry

end inequality_system_solution_l93_9371


namespace max_b_value_l93_9369

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := true

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 4

-- Define the condition for the line not passing through lattice points
def no_lattice_intersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 150 → is_lattice_point x y →
    line_equation m x ≠ y

-- State the theorem
theorem max_b_value :
  ∃ b : ℚ, b = 50/147 ∧
  (∀ m : ℚ, 1/3 < m ∧ m < b → no_lattice_intersection m) ∧
  (∀ b' : ℚ, b < b' →
    ∃ m : ℚ, 1/3 < m ∧ m < b' ∧ ¬(no_lattice_intersection m)) :=
sorry

end max_b_value_l93_9369


namespace raft_distance_l93_9328

/-- Given a motorboat that travels downstream and upstream in equal time,
    this theorem proves the distance a raft travels with the stream. -/
theorem raft_distance (t : ℝ) (vb vs : ℝ) : t > 0 →
  (vb + vs) * t = 90 →
  (vb - vs) * t = 70 →
  vs * t = 10 := by
  sorry

end raft_distance_l93_9328


namespace ellipse_sum_a_k_l93_9397

def ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

def focus1 : ℝ × ℝ := (2, 2)
def focus2 : ℝ × ℝ := (2, 6)
def point_on_ellipse : ℝ × ℝ := (-3, 4)

theorem ellipse_sum_a_k (h k a b : ℝ) :
  a > 0 → b > 0 →
  ellipse h k a b (point_on_ellipse.1) (point_on_ellipse.2) →
  (∀ x y, ellipse h k a b x y →
    Real.sqrt ((x - focus1.1)^2 + (y - focus1.2)^2) +
    Real.sqrt ((x - focus2.1)^2 + (y - focus2.2)^2) =
    Real.sqrt ((point_on_ellipse.1 - focus1.1)^2 + (point_on_ellipse.2 - focus1.2)^2) +
    Real.sqrt ((point_on_ellipse.1 - focus2.1)^2 + (point_on_ellipse.2 - focus2.2)^2)) →
  a + k = (Real.sqrt 29 + 13) / 2 := by
  sorry

end ellipse_sum_a_k_l93_9397


namespace molecular_weight_AlOH3_is_correct_l93_9388

/-- The molecular weight of Al(OH)3 -/
def molecular_weight_AlOH3 : ℝ := 78

/-- The number of moles given in the problem -/
def given_moles : ℝ := 7

/-- The total molecular weight for the given number of moles -/
def total_weight : ℝ := 546

/-- Theorem stating that the molecular weight of Al(OH)3 is correct -/
theorem molecular_weight_AlOH3_is_correct :
  molecular_weight_AlOH3 = total_weight / given_moles :=
by sorry

end molecular_weight_AlOH3_is_correct_l93_9388


namespace yogurt_combinations_l93_9304

theorem yogurt_combinations (n_flavors : ℕ) (n_toppings : ℕ) : 
  n_flavors = 4 → n_toppings = 8 → 
  n_flavors * (n_toppings.choose 3) = 224 := by
  sorry

end yogurt_combinations_l93_9304


namespace remainder_444_power_444_mod_13_l93_9329

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l93_9329


namespace mountaineering_teams_l93_9300

/-- Represents the number of teams that can be formed in a mountaineering competition. -/
def max_teams (total_students : ℕ) (advanced_climbers : ℕ) (intermediate_climbers : ℕ) (beginner_climbers : ℕ)
  (advanced_points : ℕ) (intermediate_points : ℕ) (beginner_points : ℕ)
  (team_advanced : ℕ) (team_intermediate : ℕ) (team_beginner : ℕ)
  (max_team_points : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of teams that can be formed under the given constraints. -/
theorem mountaineering_teams :
  max_teams 172 45 70 57 80 50 30 5 8 5 1000 = 8 :=
by sorry

end mountaineering_teams_l93_9300


namespace min_value_theorem_l93_9345

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 5 + 2 * Real.sqrt 6 :=
by sorry

end min_value_theorem_l93_9345


namespace jills_nickels_l93_9324

/-- Proves that Jill has 30 nickels given the conditions of the problem -/
theorem jills_nickels (total_coins : ℕ) (total_value : ℚ) (nickel_value dime_value : ℚ) :
  total_coins = 50 →
  total_value = (350 : ℚ) / 100 →
  nickel_value = (5 : ℚ) / 100 →
  dime_value = (10 : ℚ) / 100 →
  ∃ (nickels dimes : ℕ),
    nickels + dimes = total_coins ∧
    nickels * nickel_value + dimes * dime_value = total_value ∧
    nickels = 30 :=
by sorry

end jills_nickels_l93_9324


namespace imaginary_part_of_product_l93_9361

theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (3 + Complex.I)) = -2 := by
  sorry

end imaginary_part_of_product_l93_9361


namespace total_books_count_l93_9331

/-- The number of books Susan has -/
def susan_books : ℕ := 600

/-- The number of books Lidia has -/
def lidia_books : ℕ := 4 * susan_books

/-- The total number of books Susan and Lidia have -/
def total_books : ℕ := susan_books + lidia_books

theorem total_books_count : total_books = 3000 := by
  sorry

end total_books_count_l93_9331


namespace polygon_sides_from_angle_sum_l93_9325

theorem polygon_sides_from_angle_sum (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 900 → (n - 2) * 180 = sum_angles → n = 7 := by
  sorry

end polygon_sides_from_angle_sum_l93_9325


namespace max_value_of_circle_l93_9341

theorem max_value_of_circle (x y : ℝ) :
  x^2 + y^2 + 4*x - 2*y - 4 = 0 →
  x^2 + y^2 ≤ 14 + 6 * Real.sqrt 5 :=
by
  sorry

end max_value_of_circle_l93_9341


namespace prob_five_three_l93_9386

/-- Represents the probability of reaching (0,0) from a given point (x,y) -/
def P (x y : ℕ) : ℚ :=
  sorry

/-- The probability of reaching (0,0) from any point on the x-axis (except origin) is 0 -/
axiom P_x_axis (x : ℕ) : x > 0 → P x 0 = 0

/-- The probability of reaching (0,0) from any point on the y-axis (except origin) is 0 -/
axiom P_y_axis (y : ℕ) : y > 0 → P 0 y = 0

/-- The probability at the origin is 1 -/
axiom P_origin : P 0 0 = 1

/-- The recursive relation for the probability function -/
axiom P_recursive (x y : ℕ) : x > 0 → y > 0 → 
  P x y = (1/3 : ℚ) * (P (x-1) y + P x (y-1) + P (x-1) (y-1))

/-- The main theorem: probability of reaching (0,0) from (5,3) is 121/729 -/
theorem prob_five_three : P 5 3 = 121 / 729 :=
  sorry

end prob_five_three_l93_9386


namespace toms_original_portion_l93_9301

theorem toms_original_portion (tom uma vicky : ℝ) : 
  tom + uma + vicky = 2000 →
  (tom - 200) + 3 * uma + 3 * vicky = 3500 →
  tom = 1150 := by
sorry

end toms_original_portion_l93_9301


namespace next_price_reduction_l93_9336

def price_sequence (n : ℕ) : ℚ :=
  (1024 : ℚ) * (5/8 : ℚ)^n

theorem next_price_reduction : price_sequence 4 = 156.25 := by
  sorry

end next_price_reduction_l93_9336


namespace prob_one_success_in_three_trials_l93_9392

/-- The probability of exactly one success in three independent trials with success probability 3/4 -/
theorem prob_one_success_in_three_trials : 
  let p : ℚ := 3/4  -- Probability of success in each trial
  let n : ℕ := 3    -- Number of trials
  let k : ℕ := 1    -- Number of successes we're interested in
  Nat.choose n k * p^k * (1-p)^(n-k) = 9/64 := by
  sorry

end prob_one_success_in_three_trials_l93_9392


namespace jimmy_speed_l93_9339

theorem jimmy_speed (mary_speed : ℝ) (total_distance : ℝ) (time : ℝ) (jimmy_speed : ℝ) : 
  mary_speed = 5 →
  total_distance = 9 →
  time = 1 →
  jimmy_speed = total_distance - mary_speed * time →
  jimmy_speed = 4 :=
by sorry

end jimmy_speed_l93_9339


namespace data_statistics_l93_9354

def data : List ℝ := [6, 8, 8, 9, 8, 9, 8, 8, 7, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_statistics :
  mode data = 8 ∧
  median data = 8 ∧
  mean data = 8 ∧
  variance data ≠ 8 := by sorry

end data_statistics_l93_9354


namespace equation_solution_range_l93_9373

theorem equation_solution_range (x m : ℝ) : 
  (2 * x + 4 = m - x) → (x < 0) → (m < 4) := by
  sorry

end equation_solution_range_l93_9373


namespace total_pencils_count_l93_9355

/-- The number of colors in the rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of pencils in a color box -/
def pencils_per_box : ℕ := rainbow_colors

/-- The number of Emily's friends who bought a color box -/
def emilys_friends : ℕ := 7

/-- The total number of pencils Emily and her friends have -/
def total_pencils : ℕ := pencils_per_box + emilys_friends * pencils_per_box

theorem total_pencils_count : total_pencils = 56 := by
  sorry

end total_pencils_count_l93_9355


namespace cuboid_surface_area_example_l93_9391

/-- Calculates the surface area of a cuboid given its length, breadth, and height. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + length * height + breadth * height)

/-- Theorem stating that the surface area of a cuboid with length 12, breadth 14, and height 7 is 700. -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 12 14 7 = 700 := by
  sorry

end cuboid_surface_area_example_l93_9391


namespace park_fencing_cost_l93_9359

theorem park_fencing_cost (length width area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  length * width = 3750 →
  perimeter = 2 * (length + width) →
  total_cost = 175 →
  (total_cost / perimeter) * 100 = 70 :=
by sorry

end park_fencing_cost_l93_9359


namespace parabola_C_passes_through_origin_l93_9365

-- Define the parabolas
def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2*x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

-- Define what it means for a parabola to pass through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- Theorem stating that parabola C passes through the origin while others do not
theorem parabola_C_passes_through_origin :
  passes_through_origin parabola_C ∧
  ¬passes_through_origin parabola_A ∧
  ¬passes_through_origin parabola_B ∧
  ¬passes_through_origin parabola_D :=
by sorry

end parabola_C_passes_through_origin_l93_9365


namespace train_length_l93_9334

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) →
  platform_length = 240 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 280 :=
by sorry

end train_length_l93_9334


namespace square_minus_product_equals_one_l93_9343

theorem square_minus_product_equals_one (x : ℝ) : (x + 2)^2 - (x + 1) * (x + 3) = 1 := by
  sorry

end square_minus_product_equals_one_l93_9343


namespace lakers_win_in_seven_l93_9356

def probability_celtics_win : ℚ := 3/4

def probability_lakers_win : ℚ := 1 - probability_celtics_win

def games_to_win : ℕ := 4

def total_games : ℕ := 7

theorem lakers_win_in_seven (probability_celtics_win : ℚ) 
  (h1 : probability_celtics_win = 3/4) 
  (h2 : games_to_win = 4) 
  (h3 : total_games = 7) : 
  ℚ :=
by
  sorry

end lakers_win_in_seven_l93_9356


namespace average_temperature_of_three_cities_l93_9340

/-- The average temperature of three cities given specific temperature relationships --/
theorem average_temperature_of_three_cities 
  (temp_new_york : ℝ)
  (temp_diff_miami_new_york : ℝ)
  (temp_diff_san_diego_miami : ℝ)
  (h1 : temp_new_york = 80)
  (h2 : temp_diff_miami_new_york = 10)
  (h3 : temp_diff_san_diego_miami = 25) :
  (temp_new_york + (temp_new_york + temp_diff_miami_new_york) + 
   (temp_new_york + temp_diff_miami_new_york + temp_diff_san_diego_miami)) / 3 = 95 := by
  sorry

#check average_temperature_of_three_cities

end average_temperature_of_three_cities_l93_9340


namespace eight_divided_by_recurring_third_l93_9306

theorem eight_divided_by_recurring_third (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end eight_divided_by_recurring_third_l93_9306


namespace prob_ratio_balls_in_bins_l93_9357

def factorial (n : ℕ) : ℕ := sorry

def multinomial_coefficient (n : ℕ) (x : List ℕ) : ℝ := sorry

def p (n : ℕ) (k : ℕ) : ℝ := 
  multinomial_coefficient n [3, 6, 5, 4, 2, 10]

def q (n : ℕ) (k : ℕ) : ℝ := 
  multinomial_coefficient n [5, 5, 5, 5, 5, 5]

theorem prob_ratio_balls_in_bins : 
  p 30 6 / q 30 6 = 0.125 := by sorry

end prob_ratio_balls_in_bins_l93_9357


namespace smallest_solution_of_equation_l93_9330

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 40*x^2 + 400 = 0 → x ≥ -2*Real.sqrt 5 ∧ (∃ y, y^4 - 40*y^2 + 400 = 0 ∧ y = -2*Real.sqrt 5) :=
by sorry

end smallest_solution_of_equation_l93_9330


namespace largest_number_with_sum_18_l93_9347

def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 18 ∧ (n.digits 10).Nodup

theorem largest_number_with_sum_18 :
  ∀ n : ℕ, is_valid_number n → n ≤ 843210 :=
by sorry

end largest_number_with_sum_18_l93_9347


namespace right_triangle_cos_c_l93_9387

theorem right_triangle_cos_c (A B C : ℝ) (h1 : A + B + C = π) (h2 : A = π/2) (h3 : Real.sin B = 3/5) : 
  Real.cos C = 3/5 := by
  sorry

end right_triangle_cos_c_l93_9387


namespace product_of_integers_l93_9335

theorem product_of_integers (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p + q + r = 27 →
  1 / p + 1 / q + 1 / r + 300 / (p * q * r) = 1 →
  p * q * r = 984 := by
sorry

end product_of_integers_l93_9335


namespace purely_imaginary_sufficient_not_necessary_l93_9393

theorem purely_imaginary_sufficient_not_necessary (m : ℝ) :
  (∃ (z : ℂ), z = (m^2 - 1 : ℂ) + (m - 1 : ℂ) * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0) →
  (m = 1 ∨ m = -1) ∧
  ¬(∀ m : ℝ, (m = 1 ∨ m = -1) → 
    (∃ (z : ℂ), z = (m^2 - 1 : ℂ) + (m - 1 : ℂ) * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0)) :=
by sorry

end purely_imaginary_sufficient_not_necessary_l93_9393


namespace restaurant_menu_theorem_l93_9314

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem restaurant_menu_theorem (v : ℕ) : 
  (choose 5 2 * choose v 2 > 200) → v ≥ 7 := by sorry

end restaurant_menu_theorem_l93_9314


namespace quadrilateral_area_is_153_l93_9315

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculate the y-intercept of a line given its slope and a point it passes through -/
def calculateYIntercept (slope : ℝ) (p : Point) : ℝ :=
  p.y - slope * p.x

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Main theorem statement -/
theorem quadrilateral_area_is_153 (line1 : Line) (line2 : Line) (O E C : Point) : 
  line1.slope = -3 ∧ 
  E.x = 6 ∧ E.y = 6 ∧ 
  C.x = 10 ∧ C.y = 0 ∧ 
  O.x = 0 ∧ O.y = 0 ∧
  E.y = line1.slope * E.x + line1.intercept ∧
  E.y = line2.slope * E.x + line2.intercept ∧
  C.y = line2.slope * C.x + line2.intercept →
  let B : Point := { x := 0, y := calculateYIntercept line1.slope E }
  let areaOBE := triangleArea O B E
  let areaOEC := triangleArea O E C
  let areaEBC := triangleArea E B C
  areaOBE + areaOEC - areaEBC = 153 := by
  sorry

end quadrilateral_area_is_153_l93_9315


namespace sum_of_squares_first_10_base6_l93_9398

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- Computes the sum of squares of the first n base-6 numbers --/
def sumOfSquaresBase6 (n : ℕ) : ℕ := sorry

theorem sum_of_squares_first_10_base6 :
  base10ToBase6 (sumOfSquaresBase6 10) = 231 := by sorry

end sum_of_squares_first_10_base6_l93_9398


namespace sum_of_ages_l93_9320

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71 years. -/
theorem sum_of_ages (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age = 2 * shannen_age + 5 →
  beckett_age + olaf_age + shannen_age + jack_age = 71 := by
  sorry

end sum_of_ages_l93_9320


namespace trout_catfish_ratio_is_three_to_one_l93_9378

/-- Represents the fishing challenge scenario -/
structure FishingChallenge where
  will_catfish : ℕ
  will_eels : ℕ
  total_fish : ℕ

/-- Calculates the ratio of trout to catfish Henry challenged himself to catch -/
def trout_catfish_ratio (challenge : FishingChallenge) : ℚ :=
  let will_total := challenge.will_catfish + challenge.will_eels
  let henry_fish := challenge.total_fish - will_total
  (henry_fish : ℚ) / (challenge.will_catfish : ℚ) / 2

/-- Theorem stating the ratio of trout to catfish Henry challenged himself to catch -/
theorem trout_catfish_ratio_is_three_to_one (challenge : FishingChallenge)
  (h1 : challenge.will_catfish = 16)
  (h2 : challenge.will_eels = 10)
  (h3 : challenge.total_fish = 50) :
  trout_catfish_ratio challenge = 3 := by
  sorry

end trout_catfish_ratio_is_three_to_one_l93_9378


namespace amanda_candy_bars_l93_9351

/-- Amanda's candy bar problem -/
theorem amanda_candy_bars :
  let initial_bars : ℕ := 7
  let first_gift : ℕ := 3
  let new_bars : ℕ := 30
  let second_gift : ℕ := 4 * first_gift
  let kept_bars : ℕ := (initial_bars - first_gift) + (new_bars - second_gift)
  kept_bars = 22 := by sorry

end amanda_candy_bars_l93_9351


namespace base_conversion_problem_l93_9337

theorem base_conversion_problem :
  ∀ (a b : ℕ),
  (a < 10 ∧ b < 10) →
  (5 * 7^2 + 2 * 7 + 5 = 3 * 10 * a + b) →
  (a * b) / 15 = 8 / 5 := by
sorry

end base_conversion_problem_l93_9337


namespace twelfth_odd_multiple_of_5_l93_9346

/-- The nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Predicate for a number being odd and a multiple of 5 -/
def isOddMultipleOf5 (x : ℕ) : Prop :=
  x % 2 = 1 ∧ x % 5 = 0

theorem twelfth_odd_multiple_of_5 :
  nthOddMultipleOf5 12 = 115 ∧
  isOddMultipleOf5 (nthOddMultipleOf5 12) :=
sorry

end twelfth_odd_multiple_of_5_l93_9346


namespace negation_of_existence_negation_of_quadratic_equation_l93_9342

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ (∀ x : ℝ, x^2 + 2*x - 8 ≠ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_equation_l93_9342


namespace set_B_equals_l93_9370

-- Define set A
def A : Set Int := {-1, 0, 1, 2}

-- Define the function f(x) = x^2 - 2x
def f (x : Int) : Int := x^2 - 2*x

-- Define set B
def B : Set Int := {y | ∃ x ∈ A, f x = y}

-- Theorem statement
theorem set_B_equals : B = {-1, 0, 3} := by sorry

end set_B_equals_l93_9370


namespace sum_coefficients_expansion_l93_9318

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the sum of coefficients function
def sumCoefficients (x : ℕ) : ℕ :=
  (C x 1 + C (x+1) 1 + C (x+2) 1 + C (x+3) 1) ^ 2

-- Theorem statement
theorem sum_coefficients_expansion :
  ∃ x : ℕ, sumCoefficients x = 225 :=
sorry

end sum_coefficients_expansion_l93_9318


namespace total_distance_traveled_l93_9379

/-- Given the conditions of the problem, prove that the total distance traveled is 5 miles. -/
theorem total_distance_traveled (total_time : ℝ) (walking_time : ℝ) (walking_rate : ℝ) 
  (break_time : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  total_time = 75 / 60 → 
  walking_time = 1 →
  walking_rate = 3 →
  break_time = 5 / 60 →
  running_time = 1 / 6 →
  running_rate = 12 →
  walking_time * walking_rate + running_time * running_rate = 5 := by
  sorry

end total_distance_traveled_l93_9379


namespace quadratic_equation_general_form_l93_9333

theorem quadratic_equation_general_form :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 3 ∧
  ∀ x, (x - 1)^2 = 3*x - 2 ↔ a*x^2 + b*x + c = 0 :=
by sorry

end quadratic_equation_general_form_l93_9333


namespace student_D_most_stable_l93_9352

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define a function to get the variance for each student
def variance : Student → ℝ
| Student.A => 2.1
| Student.B => 3.5
| Student.C => 9.0
| Student.D => 0.7

-- Define a predicate for most stable performance
def most_stable (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

-- Theorem: Student D has the most stable performance
theorem student_D_most_stable : most_stable Student.D := by
  sorry

-- Note: The proof is omitted as per the instructions

end student_D_most_stable_l93_9352


namespace isosceles_right_triangle_area_l93_9358

theorem isosceles_right_triangle_area (DE DF : ℝ) (angle_EDF : ℝ) :
  DE = 5 →
  DF = 5 →
  angle_EDF = Real.pi / 2 →
  (1 / 2) * DE * DF = 25 / 2 := by
  sorry

end isosceles_right_triangle_area_l93_9358


namespace pet_shelter_problem_l93_9390

theorem pet_shelter_problem (total : ℕ) (apples chicken cheese : ℕ)
  (apples_chicken apples_cheese chicken_cheese : ℕ) (all_three : ℕ)
  (h_total : total = 100)
  (h_apples : apples = 20)
  (h_chicken : chicken = 70)
  (h_cheese : cheese = 10)
  (h_apples_chicken : apples_chicken = 7)
  (h_apples_cheese : apples_cheese = 3)
  (h_chicken_cheese : chicken_cheese = 5)
  (h_all_three : all_three = 2) :
  total - (apples + chicken + cheese
          - apples_chicken - apples_cheese - chicken_cheese
          + all_three) = 13 := by
  sorry

end pet_shelter_problem_l93_9390


namespace line_perpendicular_implies_plane_perpendicular_l93_9372

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (in_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_implies_plane_perpendicular
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : in_plane m α)
  (m_perp_β : perpendicular_line_plane m β) :
  perpendicular_plane_plane α β :=
sorry

end line_perpendicular_implies_plane_perpendicular_l93_9372


namespace units_digit_p_plus_4_l93_9364

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Definition of a positive even integer with a positive units digit -/
def isPositiveEvenWithPositiveUnitsDigit (p : ℕ) : Prop :=
  p > 0 ∧ p % 2 = 0 ∧ unitsDigit p > 0

/-- The main theorem -/
theorem units_digit_p_plus_4 (p : ℕ) 
  (h1 : isPositiveEvenWithPositiveUnitsDigit p)
  (h2 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 4) = 0 := by
sorry

end units_digit_p_plus_4_l93_9364


namespace robins_hair_length_l93_9305

/-- Calculates the final hair length after growth and cut -/
def final_hair_length (initial : ℕ) (growth : ℕ) (cut : ℕ) : ℕ :=
  if initial + growth ≥ cut then
    initial + growth - cut
  else
    0

/-- Theorem stating that Robin's final hair length is 2 inches -/
theorem robins_hair_length :
  final_hair_length 14 8 20 = 2 := by
  sorry

end robins_hair_length_l93_9305


namespace quadratic_function_properties_l93_9374

theorem quadratic_function_properties (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 1 3 → a = 1 ∧ b = 1) ∧
  ((a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 1) → ∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) ∧
  (∀ x, |x| ≥ 2 → f x ≥ 0) ∧
  (∀ x ∈ Set.Ioc 2 3, f x ≤ 1) ∧
  (f 3 = 1) →
  (32 : ℝ) ≤ a^2 + b^2 ∧ a^2 + b^2 ≤ 74 :=
by sorry


end quadratic_function_properties_l93_9374


namespace lighter_box_identification_l93_9362

/-- A weighing strategy for identifying a lighter box among n boxes. -/
def WeighingStrategy (n : ℕ) := ℕ

/-- The number of weighings required to identify a lighter box among n boxes. -/
def NumWeighings (strategy : WeighingStrategy n) : ℕ := sorry

/-- Checks if a strategy correctly identifies the lighter box. -/
def IsValidStrategy (strategy : WeighingStrategy n) : Prop := sorry

theorem lighter_box_identification :
  ∃ (strategy : WeighingStrategy 15),
    IsValidStrategy strategy ∧ NumWeighings strategy ≤ 4 := by sorry

end lighter_box_identification_l93_9362


namespace f_odd_and_increasing_l93_9385

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Statement to prove
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
sorry

end f_odd_and_increasing_l93_9385


namespace price_difference_enhanced_basic_computer_l93_9366

/-- Prove the price difference between enhanced and basic computers --/
theorem price_difference_enhanced_basic_computer :
  ∀ (basic_price enhanced_price printer_price : ℕ),
  basic_price = 1500 →
  basic_price + printer_price = 2500 →
  printer_price = (enhanced_price + printer_price) / 3 →
  enhanced_price - basic_price = 500 := by
  sorry

end price_difference_enhanced_basic_computer_l93_9366


namespace hyperbola_eccentricity_l93_9322

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0), if an isosceles right triangle
    MF₁F₂ is constructed with F₁ as the right-angle vertex and the midpoint of side MF₁ lies on the
    hyperbola, then the eccentricity of the hyperbola is (√5 + 1)/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)  -- Definition of eccentricity for a hyperbola
  ∃ (x y : ℝ), 
    x^2 / a^2 - y^2 / b^2 = 1 ∧  -- Point (x, y) is on the hyperbola
    x = -Real.sqrt (a^2 + b^2) / 2 ∧  -- x-coordinate of the midpoint of MF₁
    y = b^2 / (2*a) →  -- y-coordinate of the midpoint of MF₁
  e = (Real.sqrt 5 + 1) / 2 := by
sorry

end hyperbola_eccentricity_l93_9322


namespace arithmetic_evaluation_l93_9316

theorem arithmetic_evaluation : 5 + 12 / 3 - 3^2 + 1 = 1 := by
  sorry

end arithmetic_evaluation_l93_9316


namespace octagon_diagonals_l93_9319

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 vertices -/
def octagon_vertices : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_vertices = 20 := by
  sorry

end octagon_diagonals_l93_9319


namespace partner_A_money_received_l93_9312

/-- Calculates the money received by partner A in a business partnership --/
def money_received_by_A (total_profit : ℝ) : ℝ :=
  let management_share := 0.12 * total_profit
  let remaining_profit := total_profit - management_share
  let A_share_of_remaining := 0.35 * remaining_profit
  management_share + A_share_of_remaining

/-- Theorem stating that partner A receives Rs. 7062 given the problem conditions --/
theorem partner_A_money_received :
  money_received_by_A 16500 = 7062 := by
  sorry

#eval money_received_by_A 16500

end partner_A_money_received_l93_9312


namespace thomas_total_bill_l93_9394

-- Define the shipping rates
def flat_rate : ℝ := 5.00
def clothes_rate : ℝ := 0.20
def accessories_rate : ℝ := 0.10
def price_threshold : ℝ := 50.00

-- Define the prices of items
def shirt_price : ℝ := 12.00
def socks_price : ℝ := 5.00
def shorts_price : ℝ := 15.00
def swim_trunks_price : ℝ := 14.00
def hat_price : ℝ := 6.00
def sunglasses_price : ℝ := 30.00

-- Define the quantities of items
def shirt_quantity : ℕ := 3
def shorts_quantity : ℕ := 2

-- Calculate the total cost of clothes and accessories
def clothes_cost : ℝ := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity + swim_trunks_price
def accessories_cost : ℝ := hat_price + sunglasses_price

-- Calculate the shipping costs
def clothes_shipping : ℝ := clothes_rate * clothes_cost
def accessories_shipping : ℝ := accessories_rate * accessories_cost

-- Calculate the total bill
def total_bill : ℝ := clothes_cost + accessories_cost + clothes_shipping + accessories_shipping

-- Theorem to prove
theorem thomas_total_bill : total_bill = 141.60 := by sorry

end thomas_total_bill_l93_9394


namespace cups_brought_to_class_l93_9310

theorem cups_brought_to_class 
  (total_students : ℕ) 
  (num_boys : ℕ) 
  (cups_per_boy : ℕ) 
  (h1 : total_students = 30)
  (h2 : num_boys = 10)
  (h3 : cups_per_boy = 5)
  (h4 : total_students = num_boys + 2 * num_boys) :
  num_boys * cups_per_boy = 50 := by
  sorry

end cups_brought_to_class_l93_9310


namespace third_year_interest_l93_9389

/-- Calculates the compound interest for a given principal, rate, and time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Represents the loan scenario with given parameters -/
structure LoanScenario where
  initialLoan : ℝ
  rate1 : ℝ
  rate2 : ℝ
  rate3 : ℝ

/-- Theorem stating the interest paid in the third year of the loan -/
theorem third_year_interest (loan : LoanScenario) 
  (h1 : loan.initialLoan = 9000)
  (h2 : loan.rate1 = 0.09)
  (h3 : loan.rate2 = 0.105)
  (h4 : loan.rate3 = 0.085) :
  let firstYearTotal := loan.initialLoan * (1 + loan.rate1)
  let secondYearTotal := firstYearTotal * (1 + loan.rate2)
  compoundInterest secondYearTotal loan.rate3 1 = 922.18 := by
  sorry

end third_year_interest_l93_9389


namespace existence_of_functions_composition_inequality_l93_9353

-- Part 1
theorem existence_of_functions :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f (g x) = g (f x)) ∧ 
    (∀ x, f (f x) = g (g x)) ∧ 
    (∀ x, f x ≠ g x) := by sorry

-- Part 2
theorem composition_inequality 
  (f₁ g₁ : ℝ → ℝ) 
  (h₁ : ∀ x, f₁ (g₁ x) = g₁ (f₁ x)) 
  (h₂ : ∀ x, f₁ x ≠ g₁ x) : 
  ∀ x, f₁ (f₁ x) ≠ g₁ (g₁ x) := by sorry

end existence_of_functions_composition_inequality_l93_9353


namespace total_tagged_numbers_l93_9380

def card_sum (w x y z : ℕ) : ℕ := w + x + y + z

theorem total_tagged_numbers : ∃ (w x y z : ℕ),
  w = 200 ∧
  x = w / 2 ∧
  y = x + w ∧
  z = 400 ∧
  card_sum w x y z = 1000 := by
sorry

end total_tagged_numbers_l93_9380


namespace five_in_range_of_f_l93_9360

/-- The function f(x) = x^2 + bx - 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 3

/-- Theorem stating that 5 is always in the range of f(x) for all real b -/
theorem five_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 5 := by
  sorry

end five_in_range_of_f_l93_9360
