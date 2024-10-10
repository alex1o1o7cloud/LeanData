import Mathlib

namespace intersection_with_complement_l2675_267549

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_with_complement : A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end intersection_with_complement_l2675_267549


namespace cakes_sold_minus_bought_l2675_267598

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 274. -/
theorem cakes_sold_minus_bought (initial : ℕ) (sold : ℕ) (bought : ℕ) 
    (h1 : initial = 648) 
    (h2 : sold = 467) 
    (h3 : bought = 193) : 
    sold - bought = 274 := by
  sorry

end cakes_sold_minus_bought_l2675_267598


namespace cube_volume_problem_l2675_267587

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →  -- Ensure a is positive for a valid cube
  (a^3 - ((a + 1)^2 * (a - 2)) = 10) → 
  (a^3 = 216) := by
sorry

end cube_volume_problem_l2675_267587


namespace fixed_point_of_exponential_function_l2675_267593

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end fixed_point_of_exponential_function_l2675_267593


namespace christophers_age_l2675_267506

/-- Proves Christopher's age given the conditions of the problem -/
theorem christophers_age (christopher george ford : ℕ) 
  (h1 : george = christopher + 8)
  (h2 : ford = christopher - 2)
  (h3 : christopher + george + ford = 60) :
  christopher = 18 := by
  sorry

end christophers_age_l2675_267506


namespace product_is_2008th_power_l2675_267523

theorem product_is_2008th_power : ∃ (a b c n : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = (a + c) / 2 ∧
  a * b * c = n^2008 := by
sorry

end product_is_2008th_power_l2675_267523


namespace monthly_earnings_calculation_l2675_267552

/-- Proves that a person with given savings and earnings parameters earns a specific monthly amount -/
theorem monthly_earnings_calculation (savings_per_month : ℕ) 
                                     (car_cost : ℕ) 
                                     (total_earnings : ℕ) 
                                     (h1 : savings_per_month = 500)
                                     (h2 : car_cost = 45000)
                                     (h3 : total_earnings = 360000) : 
  (total_earnings / (car_cost / savings_per_month) : ℚ) = 4000 := by
  sorry

end monthly_earnings_calculation_l2675_267552


namespace team_selection_count_l2675_267567

/-- The number of ways to select a team of 5 members from a group of 7 boys and 9 girls, 
    with at least 2 boys in the team -/
def select_team (num_boys num_girls : ℕ) : ℕ :=
  (num_boys.choose 2 * num_girls.choose 3) +
  (num_boys.choose 3 * num_girls.choose 2) +
  (num_boys.choose 4 * num_girls.choose 1) +
  (num_boys.choose 5 * num_girls.choose 0)

/-- Theorem stating that the number of ways to select the team is 3360 -/
theorem team_selection_count :
  select_team 7 9 = 3360 := by
  sorry

end team_selection_count_l2675_267567


namespace august_tips_multiple_l2675_267528

theorem august_tips_multiple (total_months : ℕ) (other_months : ℕ) (august_ratio : ℝ) :
  total_months = 7 →
  other_months = 6 →
  august_ratio = 0.5714285714285714 →
  ∃ (avg_other_months : ℝ),
    avg_other_months > 0 →
    august_ratio * (8 * avg_other_months + other_months * avg_other_months) = 8 * avg_other_months :=
by sorry

end august_tips_multiple_l2675_267528


namespace sum_of_squares_l2675_267503

theorem sum_of_squares (x y z : ℕ+) : 
  (x : ℕ) + y + z = 24 →
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  ∃! s : ℕ, s = x^2 + y^2 + z^2 ∧ s = 296 := by
sorry

end sum_of_squares_l2675_267503


namespace rubiks_cube_purchase_l2675_267518

theorem rubiks_cube_purchase (price_A price_B total_cubes max_funding : ℕ)
  (h1 : price_A = 15)
  (h2 : price_B = 22)
  (h3 : total_cubes = 40)
  (h4 : max_funding = 776) :
  ∃ (x : ℕ), x = 15 ∧
    x ≤ total_cubes - x ∧
    x * price_A + (total_cubes - x) * price_B ≤ max_funding ∧
    ∀ (y : ℕ), y < x →
      y > total_cubes - y ∨
      y * price_A + (total_cubes - y) * price_B > max_funding :=
by sorry

end rubiks_cube_purchase_l2675_267518


namespace circles_externally_tangent_l2675_267534

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) = r₁ + r₂

theorem circles_externally_tangent :
  let c₁ : ℝ × ℝ := (0, 8)
  let c₂ : ℝ × ℝ := (-6, 0)
  let r₁ : ℝ := 6
  let r₂ : ℝ := 2
  externally_tangent c₁ c₂ r₁ r₂ := by
  sorry

#check circles_externally_tangent

end circles_externally_tangent_l2675_267534


namespace percentage_both_correct_l2675_267546

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) : 
  p_first = 0.63 → p_second = 0.50 → p_neither = 0.20 → 
  p_first + p_second - (1 - p_neither) = 0.33 := by
  sorry

end percentage_both_correct_l2675_267546


namespace sqrt_22_properties_l2675_267502

theorem sqrt_22_properties (h : 4 < Real.sqrt 22 ∧ Real.sqrt 22 < 5) :
  (∃ (i : ℤ) (d : ℝ), i = 4 ∧ d = Real.sqrt 22 - 4 ∧ Real.sqrt 22 = i + d) ∧
  (∃ (m n : ℝ), 
    m = 7 - Real.sqrt 22 - Int.floor (7 - Real.sqrt 22) ∧
    n = 7 + Real.sqrt 22 - Int.floor (7 + Real.sqrt 22) ∧
    m + n = 1) :=
by sorry

end sqrt_22_properties_l2675_267502


namespace quilt_material_requirement_l2675_267521

theorem quilt_material_requirement (material_per_quilt : ℝ) : 
  (7 * material_per_quilt = 21) ∧ (12 * material_per_quilt = 36) :=
by sorry

end quilt_material_requirement_l2675_267521


namespace triangle_acute_obtuse_characterization_l2675_267505

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

def Triangle.isAcute (t : Triangle) : Prop :=
  t.A < Real.pi / 2 ∧ t.B < Real.pi / 2 ∧ t.C < Real.pi / 2

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

theorem triangle_acute_obtuse_characterization (t : Triangle) :
  (t.isAcute ↔ Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 < 1) ∧
  (t.isObtuse ↔ Real.cos t.A ^ 2 + Real.cos t.B ^ 2 + Real.cos t.C ^ 2 > 1) :=
sorry

end triangle_acute_obtuse_characterization_l2675_267505


namespace common_chord_theorem_l2675_267586

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 3*x - 3*y + 3 = 0

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

/-- The equation of the line containing the common chord -/
def common_chord_line (x y : ℝ) : Prop := x + y - 3 = 0

/-- Theorem stating the equation of the common chord and its length -/
theorem common_chord_theorem :
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord_line x y) ∧
  (∃ a b c d : ℝ, C₁ a b ∧ C₂ a b ∧ C₁ c d ∧ C₂ c d ∧
    common_chord_line a b ∧ common_chord_line c d ∧
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 6^(1/2 : ℝ)) :=
by sorry

end common_chord_theorem_l2675_267586


namespace units_digit_of_1583_pow_1246_l2675_267550

theorem units_digit_of_1583_pow_1246 : ∃ n : ℕ, 1583^1246 ≡ 9 [ZMOD 10] :=
by sorry

end units_digit_of_1583_pow_1246_l2675_267550


namespace red_balls_estimate_l2675_267511

/-- Estimates the number of red balls in a bag given the total number of balls,
    the number of draws, and the number of red balls drawn. -/
def estimate_red_balls (total_balls : ℕ) (total_draws : ℕ) (red_draws : ℕ) : ℕ :=
  (total_balls * red_draws) / total_draws

/-- Theorem stating that under the given conditions, the estimated number of red balls is 6. -/
theorem red_balls_estimate :
  let total_balls : ℕ := 20
  let total_draws : ℕ := 100
  let red_draws : ℕ := 30
  estimate_red_balls total_balls total_draws red_draws = 6 := by
  sorry

#eval estimate_red_balls 20 100 30

end red_balls_estimate_l2675_267511


namespace smallest_N_value_l2675_267522

/-- Represents a point in the rectangular array -/
structure Point where
  row : Nat
  col : Nat

/-- The original numbering function -/
def original_number (N : Nat) (p : Point) : Nat :=
  (p.row - 1) * N + p.col

/-- The new numbering function -/
def new_number (p : Point) : Nat :=
  5 * (p.col - 1) + p.row

/-- The theorem stating the smallest possible value of N -/
theorem smallest_N_value : ∃ (N : Nat) (p₁ p₂ p₃ p₄ p₅ : Point),
  N > 0 ∧
  p₁.row = 1 ∧ p₂.row = 2 ∧ p₃.row = 3 ∧ p₄.row = 4 ∧ p₅.row = 5 ∧
  p₁.col ≤ N ∧ p₂.col ≤ N ∧ p₃.col ≤ N ∧ p₄.col ≤ N ∧ p₅.col ≤ N ∧
  original_number N p₁ = new_number p₂ ∧
  original_number N p₂ = new_number p₁ ∧
  original_number N p₃ = new_number p₄ ∧
  original_number N p₄ = new_number p₅ ∧
  original_number N p₅ = new_number p₃ ∧
  (∀ (M : Nat) (q₁ q₂ q₃ q₄ q₅ : Point),
    M > 0 ∧
    q₁.row = 1 ∧ q₂.row = 2 ∧ q₃.row = 3 ∧ q₄.row = 4 ∧ q₅.row = 5 ∧
    q₁.col ≤ M ∧ q₂.col ≤ M ∧ q₃.col ≤ M ∧ q₄.col ≤ M ∧ q₅.col ≤ M ∧
    original_number M q₁ = new_number q₂ ∧
    original_number M q₂ = new_number q₁ ∧
    original_number M q₃ = new_number q₄ ∧
    original_number M q₄ = new_number q₅ ∧
    original_number M q₅ = new_number q₃ →
    M ≥ N) ∧
  N = 149 := by
  sorry

end smallest_N_value_l2675_267522


namespace equation_solution_l2675_267563

theorem equation_solution : ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) ∧ x = 9 := by
  sorry

end equation_solution_l2675_267563


namespace total_boys_in_camp_l2675_267540

theorem total_boys_in_camp (total : ℕ) 
  (h1 : (total : ℚ) * (1 / 5) = (total : ℚ) * (20 / 100))
  (h2 : (total : ℚ) * (1 / 5) * (3 / 10) = (total : ℚ) * (1 / 5) * (30 / 100))
  (h3 : (total : ℚ) * (1 / 5) * (7 / 10) = 77) :
  total = 550 := by
sorry

end total_boys_in_camp_l2675_267540


namespace range_of_a_for_absolute_value_equation_l2675_267512

theorem range_of_a_for_absolute_value_equation (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ y : ℝ, y > 0 → |y| ≠ a * y + 1) → 
  a > -1 :=
sorry

end range_of_a_for_absolute_value_equation_l2675_267512


namespace ceiling_negative_three_point_seven_l2675_267545

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_three_point_seven_l2675_267545


namespace industrial_lubricants_allocation_l2675_267564

theorem industrial_lubricants_allocation :
  let total_degrees : ℝ := 360
  let total_percentage : ℝ := 100
  let microphotonics : ℝ := 14
  let home_electronics : ℝ := 24
  let food_additives : ℝ := 15
  let genetically_modified_microorganisms : ℝ := 19
  let astrophysics_degrees : ℝ := 72
  let known_sectors := microphotonics + home_electronics + food_additives + genetically_modified_microorganisms
  let astrophysics_percentage := (astrophysics_degrees / total_degrees) * total_percentage
  let industrial_lubricants := total_percentage - known_sectors - astrophysics_percentage
  industrial_lubricants = 8 := by
sorry

end industrial_lubricants_allocation_l2675_267564


namespace choose_captains_l2675_267526

theorem choose_captains (n k : ℕ) (hn : n = 15) (hk : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end choose_captains_l2675_267526


namespace eighth_grade_girls_count_l2675_267571

theorem eighth_grade_girls_count :
  ∀ (N : ℕ), 
  (N > 0) →
  (∃ (boys girls : ℕ), 
    N = boys + girls ∧
    boys = girls + 1 ∧
    boys = (52 * N) / 100) →
  ∃ (girls : ℕ), girls = 12 :=
by sorry

end eighth_grade_girls_count_l2675_267571


namespace intersection_of_A_and_B_l2675_267500

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -4 * p.1 + 6}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 5 * p.1 - 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by
  sorry

end intersection_of_A_and_B_l2675_267500


namespace unique_integer_solution_quadratic_l2675_267584

theorem unique_integer_solution_quadratic :
  ∃! a : ℤ, ∃ x : ℤ, x^2 + a*x + a^2 = 0 := by
  sorry

end unique_integer_solution_quadratic_l2675_267584


namespace largest_number_digit_sum_l2675_267517

def digits : Finset ℕ := {5, 6, 4, 7}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_number_digit_sum :
  ∃ (max_n : ℕ), is_valid_number max_n ∧
  ∀ (n : ℕ), is_valid_number n → n ≤ max_n ∧
  digit_sum max_n = 18 :=
sorry

end largest_number_digit_sum_l2675_267517


namespace worker_b_completion_time_l2675_267560

/-- The time it takes for three workers to complete a job together and individually -/
def JobCompletion (t_together t_a t_b t_c : ℝ) : Prop :=
  (1 / t_together) = (1 / t_a) + (1 / t_b) + (1 / t_c)

/-- Theorem stating that given the conditions, worker B completes the job in 6 days -/
theorem worker_b_completion_time :
  ∀ (t_together : ℝ),
  t_together = 3.428571428571429 →
  JobCompletion t_together 24 6 12 :=
by sorry

end worker_b_completion_time_l2675_267560


namespace kiwi_weight_l2675_267565

theorem kiwi_weight (total_weight : ℝ) (apple_weight : ℝ) (orange_percent : ℝ) (strawberry_kiwi_percent : ℝ) (strawberry_kiwi_ratio : ℝ) :
  total_weight = 400 →
  apple_weight = 80 →
  orange_percent = 0.15 →
  strawberry_kiwi_percent = 0.40 →
  strawberry_kiwi_ratio = 3 / 5 →
  ∃ kiwi_weight : ℝ,
    kiwi_weight = 100 ∧
    kiwi_weight + (strawberry_kiwi_ratio * kiwi_weight) = strawberry_kiwi_percent * total_weight ∧
    kiwi_weight + (strawberry_kiwi_ratio * kiwi_weight) + (orange_percent * total_weight) + apple_weight = total_weight :=
by
  sorry

end kiwi_weight_l2675_267565


namespace no_mems_are_veens_l2675_267520

universe u

def Mem : Type u := sorry
def En : Type u := sorry
def Veen : Type u := sorry

variable (is_mem : Mem → Prop)
variable (is_en : En → Prop)
variable (is_veen : Veen → Prop)

axiom all_mems_are_ens : ∀ (m : Mem), ∃ (e : En), is_mem m → is_en e
axiom no_ens_are_veens : ¬∃ (e : En) (v : Veen), is_en e ∧ is_veen v

theorem no_mems_are_veens : ¬∃ (m : Mem) (v : Veen), is_mem m ∧ is_veen v := by
  sorry

end no_mems_are_veens_l2675_267520


namespace two_digit_number_difference_l2675_267525

/-- Given a two-digit number where the difference between its digits is 9,
    prove that the difference between the original number and the number
    with interchanged digits is always 81. -/
theorem two_digit_number_difference (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 9 →
  (10 * x + y) - (10 * y + x) = 81 := by
sorry

end two_digit_number_difference_l2675_267525


namespace helium_lowest_liquefaction_temp_l2675_267544

-- Define the gases
inductive Gas : Type
| Oxygen
| Hydrogen
| Nitrogen
| Helium

-- Define the liquefaction temperature function
def liquefaction_temp : Gas → ℝ
| Gas.Oxygen => -183
| Gas.Hydrogen => -253
| Gas.Nitrogen => -195.8
| Gas.Helium => -268

-- Statement to prove
theorem helium_lowest_liquefaction_temp :
  ∀ g : Gas, liquefaction_temp Gas.Helium ≤ liquefaction_temp g :=
by sorry

end helium_lowest_liquefaction_temp_l2675_267544


namespace fraction_simplification_l2675_267513

theorem fraction_simplification (x y z : ℝ) (h : x + 2*y + z ≠ 0) :
  (x^2 + y^2 - 4*z^2 + 2*x*y) / (x^2 + 4*y^2 - z^2 + 2*x*z) = (x + y - 2*z) / (x + z - 2*y) :=
by sorry

end fraction_simplification_l2675_267513


namespace sum_of_irrationals_not_always_irrational_student_claim_incorrect_l2675_267555

theorem sum_of_irrationals_not_always_irrational :
  ∃ (a b : ℝ), 
    (¬ ∃ (q : ℚ), a = ↑q) ∧ 
    (¬ ∃ (q : ℚ), b = ↑q) ∧ 
    (∃ (q : ℚ), a + b = ↑q) :=
by sorry

-- Given conditions
axiom sqrt_2_irrational : ¬ ∃ (q : ℚ), Real.sqrt 2 = ↑q
axiom sqrt_3_irrational : ¬ ∃ (q : ℚ), Real.sqrt 3 = ↑q
axiom sum_sqrt_2_3_irrational : ¬ ∃ (q : ℚ), Real.sqrt 2 + Real.sqrt 3 = ↑q

-- The statement to be proved
theorem student_claim_incorrect : 
  ¬ (∀ (a b : ℝ), (¬ ∃ (q : ℚ), a = ↑q) → (¬ ∃ (q : ℚ), b = ↑q) → (¬ ∃ (q : ℚ), a + b = ↑q)) :=
by sorry

end sum_of_irrationals_not_always_irrational_student_claim_incorrect_l2675_267555


namespace probability_same_color_is_one_third_l2675_267538

/-- The set of available colors for sportswear -/
inductive Color
  | Red
  | White
  | Blue

/-- The probability of two athletes choosing the same color from three options -/
def probability_same_color : ℚ :=
  1 / 3

/-- Theorem stating that the probability of two athletes choosing the same color is 1/3 -/
theorem probability_same_color_is_one_third :
  probability_same_color = 1 / 3 := by
  sorry

end probability_same_color_is_one_third_l2675_267538


namespace bells_sync_theorem_l2675_267583

/-- The time in minutes when all bells ring together -/
def bell_sync_time : ℕ := 360

/-- Periods of bell ringing for each institution in minutes -/
def museum_period : ℕ := 18
def library_period : ℕ := 24
def town_hall_period : ℕ := 30
def hospital_period : ℕ := 36

theorem bells_sync_theorem :
  bell_sync_time = Nat.lcm museum_period (Nat.lcm library_period (Nat.lcm town_hall_period hospital_period)) ∧
  bell_sync_time % museum_period = 0 ∧
  bell_sync_time % library_period = 0 ∧
  bell_sync_time % town_hall_period = 0 ∧
  bell_sync_time % hospital_period = 0 :=
by sorry

end bells_sync_theorem_l2675_267583


namespace correct_operation_l2675_267577

theorem correct_operation (x y : ℝ) : 
  (2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y) ∧ 
  (x^3 * x^5 ≠ x^15) ∧ 
  (2 * x + 3 * y ≠ 5 * x * y) ∧ 
  ((x - 2)^2 ≠ x^2 - 4) := by
  sorry

end correct_operation_l2675_267577


namespace candy_distribution_convergence_l2675_267508

/-- Represents the state of candy distribution among students -/
structure CandyState where
  numStudents : Nat
  candies : Fin numStudents → Nat

/-- Represents one round of candy distribution -/
def distributeCandy (state : CandyState) : CandyState :=
  sorry

/-- The teacher gives one candy to students with an odd number of candies -/
def teacherIntervention (state : CandyState) : CandyState :=
  sorry

/-- Checks if all students have the same number of candies -/
def allEqual (state : CandyState) : Bool :=
  sorry

/-- Main theorem: After a finite number of rounds, all students will have the same number of candies -/
theorem candy_distribution_convergence
  (initialState : CandyState)
  (h_even_initial : ∀ i, Even (initialState.candies i)) :
  ∃ n : Nat, allEqual (((teacherIntervention ∘ distributeCandy)^[n]) initialState) = true :=
sorry

end candy_distribution_convergence_l2675_267508


namespace mario_garden_blossoms_l2675_267548

/-- Calculates the total number of blossoms in Mario's garden after a given number of weeks. -/
def total_blossoms (weeks : ℕ) : ℕ :=
  let hibiscus1 := 2 + 3 * weeks
  let hibiscus2 := 4 + 4 * weeks
  let hibiscus3 := 16 + 5 * weeks
  let rose1 := 3 + 2 * weeks
  let rose2 := 5 + 3 * weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2

/-- Theorem stating that the total number of blossoms in Mario's garden after 2 weeks is 64. -/
theorem mario_garden_blossoms : total_blossoms 2 = 64 := by
  sorry

end mario_garden_blossoms_l2675_267548


namespace convex_polygon_30_sides_diagonals_l2675_267531

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end convex_polygon_30_sides_diagonals_l2675_267531


namespace line_slope_product_l2675_267510

theorem line_slope_product (m n : ℝ) (h1 : m ≠ 0) (h2 : m = 4 * n) 
  (h3 : Real.arctan m = 2 * Real.arctan n) : m * n = 2 := by
  sorry

end line_slope_product_l2675_267510


namespace geometric_sequence_property_l2675_267543

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 1 * a 4 * a 7 = 27) : 
  a 3 * a 5 = 9 := by
sorry

end geometric_sequence_property_l2675_267543


namespace sqrt_equation_solution_l2675_267554

theorem sqrt_equation_solution (x : ℝ) (h : x > 0) : Real.sqrt ((3 / x) + 3) = 2 → x = 3 := by
  sorry

end sqrt_equation_solution_l2675_267554


namespace system_solution_l2675_267559

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : y + z = b) 
  (eq3 : z + x = c) : 
  x = (a + c - b) / 2 ∧ 
  y = (a + b - c) / 2 ∧ 
  z = (b + c - a) / 2 := by
sorry


end system_solution_l2675_267559


namespace trapezoid_perimeter_l2675_267536

/-- Represents a trapezoid EFGH with point X dividing the longer base EH -/
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  EX : ℝ
  XH : ℝ

/-- The perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.FG + t.GH + (t.EX + t.XH)

/-- Theorem stating that the perimeter of the given trapezoid is 165 units -/
theorem trapezoid_perimeter : 
  ∀ t : Trapezoid, 
    t.EF = 45 ∧ 
    t.FG = 40 ∧ 
    t.GH = 35 ∧ 
    t.EX = 30 ∧ 
    t.XH = 15 → 
    perimeter t = 165 := by
  sorry


end trapezoid_perimeter_l2675_267536


namespace cubic_km_to_cubic_m_l2675_267576

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The number of cubic kilometers to convert -/
def cubic_km : ℝ := 5

/-- Theorem stating that 5 cubic kilometers is equal to 5,000,000,000 cubic meters -/
theorem cubic_km_to_cubic_m : 
  cubic_km * (km_to_m ^ 3) = 5000000000 := by
  sorry

end cubic_km_to_cubic_m_l2675_267576


namespace cone_volume_from_sector_l2675_267595

/-- The volume of a right circular cone formed by rolling up a two-third sector of a circle -/
theorem cone_volume_from_sector (r : ℝ) (h : r = 6) :
  let sector_angle : ℝ := 2 * π * (2/3)
  let base_circumference : ℝ := sector_angle * r / (2 * π)
  let base_radius : ℝ := base_circumference / (2 * π)
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = (32/3) * π * Real.sqrt 5 := by
  sorry

end cone_volume_from_sector_l2675_267595


namespace topsoil_cost_l2675_267519

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 3

/-- The total cost of topsoil in dollars -/
def total_cost : ℝ := cubic_yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem topsoil_cost : total_cost = 648 := by
  sorry

end topsoil_cost_l2675_267519


namespace nested_radical_inequality_l2675_267551

theorem nested_radical_inequality (x : ℝ) (hx : x > 0) :
  Real.sqrt (2 * x * Real.sqrt ((2 * x + 1) * Real.sqrt ((2 * x + 2) * Real.sqrt (2 * x + 3)))) < (15 * x + 6) / 8 := by
  sorry

end nested_radical_inequality_l2675_267551


namespace vasya_numbers_l2675_267515

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end vasya_numbers_l2675_267515


namespace book_ratio_is_one_fifth_l2675_267585

/-- The ratio of Queen's extra books to Alannah's books -/
def book_ratio (beatrix alannah queen total : ℕ) : ℚ :=
  let queen_extra := queen - alannah
  ↑queen_extra / ↑alannah

theorem book_ratio_is_one_fifth 
  (beatrix alannah queen total : ℕ) 
  (h1 : beatrix = 30)
  (h2 : alannah = beatrix + 20)
  (h3 : total = beatrix + alannah + queen)
  (h4 : total = 140) :
  book_ratio beatrix alannah queen total = 1 / 5 := by
  sorry

end book_ratio_is_one_fifth_l2675_267585


namespace sector_central_angle_l2675_267569

/-- Given a sector with circumference 8 and area 4, prove that its central angle is 2 radians -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) 
  (h_circumference : r * θ + 2 * r = 8) 
  (h_area : (1/2) * r^2 * θ = 4) : 
  θ = 2 := by
  sorry

end sector_central_angle_l2675_267569


namespace item_list_price_l2675_267527

/-- The list price of an item -/
def list_price : ℝ := 40

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate -/
def alice_rate : ℝ := 0.15

/-- Bob's commission rate -/
def bob_rate : ℝ := 0.25

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem item_list_price :
  alice_commission list_price = bob_commission list_price :=
by sorry

end item_list_price_l2675_267527


namespace square_sum_value_l2675_267537

theorem square_sum_value (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 := by
  sorry

end square_sum_value_l2675_267537


namespace complex_sum_magnitude_l2675_267539

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = -3) : 
  Complex.abs (a + b + c) = 1 := by
  sorry

end complex_sum_magnitude_l2675_267539


namespace log_z_w_value_l2675_267509

theorem log_z_w_value (x y z w : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) (hw : w > 0)
  (hlogx : Real.log w / Real.log x = 24)
  (hlogy : Real.log w / Real.log y = 40)
  (hlogxyz : Real.log w / Real.log (x * y * z) = 12) :
  Real.log w / Real.log z = 60 := by
  sorry

end log_z_w_value_l2675_267509


namespace add_preserves_inequality_l2675_267590

theorem add_preserves_inequality (x y : ℝ) (h : x < y) : x + 6 < y + 6 := by
  sorry

end add_preserves_inequality_l2675_267590


namespace movie_night_kernels_calculation_l2675_267529

/-- Represents a person's popcorn preference --/
structure PopcornPreference where
  name : String
  cups_wanted : ℚ
  kernel_tablespoons : ℚ
  popcorn_cups : ℚ

/-- Calculates the tablespoons of kernels needed for a given preference --/
def kernels_needed (pref : PopcornPreference) : ℚ :=
  pref.kernel_tablespoons * (pref.cups_wanted / pref.popcorn_cups)

/-- The list of popcorn preferences for the movie night --/
def movie_night_preferences : List PopcornPreference := [
  { name := "Joanie", cups_wanted := 3, kernel_tablespoons := 3, popcorn_cups := 6 },
  { name := "Mitchell", cups_wanted := 4, kernel_tablespoons := 2, popcorn_cups := 4 },
  { name := "Miles and Davis", cups_wanted := 6, kernel_tablespoons := 4, popcorn_cups := 8 },
  { name := "Cliff", cups_wanted := 3, kernel_tablespoons := 1, popcorn_cups := 3 }
]

/-- The total tablespoons of kernels needed for the movie night --/
def total_kernels_needed : ℚ :=
  movie_night_preferences.map kernels_needed |>.sum

theorem movie_night_kernels_calculation :
  total_kernels_needed = 15/2 := by
  sorry

#eval total_kernels_needed

end movie_night_kernels_calculation_l2675_267529


namespace product_of_numbers_with_sum_and_difference_l2675_267568

theorem product_of_numbers_with_sum_and_difference 
  (x y : ℝ) (sum_eq : x + y = 120) (diff_eq : x - y = 6) : x * y = 3591 := by
  sorry

end product_of_numbers_with_sum_and_difference_l2675_267568


namespace window_purchase_savings_l2675_267597

/-- Represents the window purchase scenario --/
structure WindowPurchase where
  regularPrice : ℕ
  freeWindows : ℕ
  purchaseThreshold : ℕ
  daveNeeds : ℕ
  dougNeeds : ℕ

/-- Calculates the cost for a given number of windows --/
def calculateCost (wp : WindowPurchase) (windows : ℕ) : ℕ :=
  let freeGroups := windows / wp.purchaseThreshold
  let paidWindows := windows - (freeGroups * wp.freeWindows)
  paidWindows * wp.regularPrice

/-- Calculates the savings when purchasing together vs separately --/
def calculateSavings (wp : WindowPurchase) : ℕ :=
  let separateCost := calculateCost wp wp.daveNeeds + calculateCost wp wp.dougNeeds
  let jointCost := calculateCost wp (wp.daveNeeds + wp.dougNeeds)
  separateCost - jointCost

/-- The main theorem stating the savings amount --/
theorem window_purchase_savings (wp : WindowPurchase) 
  (h1 : wp.regularPrice = 120)
  (h2 : wp.freeWindows = 2)
  (h3 : wp.purchaseThreshold = 6)
  (h4 : wp.daveNeeds = 12)
  (h5 : wp.dougNeeds = 9) :
  calculateSavings wp = 360 := by
  sorry


end window_purchase_savings_l2675_267597


namespace count_lattice_points_l2675_267514

/-- The number of lattice points on the graph of x^2 - y^2 = 36 -/
def lattice_points_count : ℕ := 8

/-- A predicate that checks if a pair of integers satisfies x^2 - y^2 = 36 -/
def satisfies_equation (x y : ℤ) : Prop := x^2 - y^2 = 36

/-- The theorem stating that there are exactly 8 lattice points on the graph of x^2 - y^2 = 36 -/
theorem count_lattice_points :
  (∃! (s : Finset (ℤ × ℤ)), s.card = lattice_points_count ∧ 
    ∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_equation p.1 p.2) :=
by sorry

#check count_lattice_points

end count_lattice_points_l2675_267514


namespace max_sum_of_squares_l2675_267580

theorem max_sum_of_squares (a b : ℝ) 
  (h : Real.sqrt ((a - 1)^2) + Real.sqrt ((a - 6)^2) = 10 - |b + 3| - |b - 2|) : 
  (∀ x y : ℝ, Real.sqrt ((x - 1)^2) + Real.sqrt ((x - 6)^2) = 10 - |y + 3| - |y - 2| → 
    x^2 + y^2 ≤ a^2 + b^2) → 
  a^2 + b^2 = 45 :=
sorry

end max_sum_of_squares_l2675_267580


namespace find_number_l2675_267573

theorem find_number : ∃ x : ℤ, x + 12 - 27 = 24 ∧ x = 39 := by
  sorry

end find_number_l2675_267573


namespace picture_distribution_l2675_267582

theorem picture_distribution (total : ℕ) (first_album : ℕ) (num_albums : ℕ) :
  total = 35 →
  first_album = 14 →
  num_albums = 3 →
  (total - first_album) % num_albums = 0 →
  (total - first_album) / num_albums = 7 := by
  sorry

end picture_distribution_l2675_267582


namespace envelope_addressing_equation_l2675_267542

theorem envelope_addressing_equation (x : ℝ) : x > 0 → (
  let rate1 := 500 / 8
  let rate2 := 500 / x
  let combined_rate := 500 / 2
  rate1 + rate2 = combined_rate
) ↔ 1/8 + 1/x = 1/2 := by
  sorry

end envelope_addressing_equation_l2675_267542


namespace geometric_sequence_general_term_l2675_267553

/-- A geometric sequence with positive terms, where a₁ = 1 and a₁ + a₂ + a₃ = 7 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  a 1 = 1 ∧
  a 1 + a 2 + a 3 = 7

/-- The general term of the geometric sequence is 2^(n-1) -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) := by
  sorry

end geometric_sequence_general_term_l2675_267553


namespace expression_simplification_l2675_267591

theorem expression_simplification (x y : ℝ) (hx : x = 4) (hy : y = -2) :
  1 - (x - y) / (x + 2*y) / ((x^2 - y^2) / (x^2 + 4*x*y + 4*y^2)) = 1 := by
  sorry

end expression_simplification_l2675_267591


namespace num_male_students_l2675_267570

/-- Proves the number of male students in an algebra test given certain conditions -/
theorem num_male_students (total_avg : ℝ) (male_avg : ℝ) (female_avg : ℝ) (num_female : ℕ) :
  total_avg = 90 →
  male_avg = 87 →
  female_avg = 92 →
  num_female = 12 →
  ∃ (num_male : ℕ),
    num_male = 8 ∧
    (num_male : ℝ) * male_avg + (num_female : ℝ) * female_avg = (num_male + num_female : ℝ) * total_avg :=
by sorry

end num_male_students_l2675_267570


namespace probability_two_heads_in_three_flips_l2675_267530

/-- A fair coin has an equal probability of landing heads or tails. -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The number of coin flips. -/
def num_flips : ℕ := 3

/-- The number of desired heads. -/
def num_heads : ℕ := 2

/-- The probability of getting exactly k successes in n trials with probability p of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

/-- The main theorem: the probability of getting exactly 2 heads in 3 flips of a fair coin is 0.375. -/
theorem probability_two_heads_in_three_flips (p : ℝ) (h : fair_coin p) :
  binomial_probability num_flips num_heads p = 0.375 := by
  sorry

end probability_two_heads_in_three_flips_l2675_267530


namespace restocking_theorem_l2675_267504

/-- Calculates the amount of ingredients needed to restock --/
def ingredients_to_buy (initial_flour initial_sugar initial_chips : ℕ)
                       (mon_flour mon_sugar mon_chips : ℕ)
                       (tue_flour tue_sugar tue_chips : ℕ)
                       (wed_flour wed_chips : ℕ)
                       (full_flour full_sugar full_chips : ℕ) :
                       (ℕ × ℕ × ℕ) :=
  let remaining_flour := initial_flour - mon_flour - tue_flour
  let spilled_flour := remaining_flour / 2
  let final_flour := if spilled_flour > wed_flour then spilled_flour - wed_flour else 0
  let flour_to_buy := full_flour + (if spilled_flour > wed_flour then 0 else wed_flour - spilled_flour)
  let sugar_to_buy := full_sugar - (initial_sugar - mon_sugar - tue_sugar)
  let chips_to_buy := full_chips - (initial_chips - mon_chips - tue_chips - wed_chips)
  (flour_to_buy, sugar_to_buy, chips_to_buy)

theorem restocking_theorem :
  ingredients_to_buy 500 300 400 150 120 200 240 90 150 100 90 500 300 400 = (545, 210, 440) := by
  sorry

end restocking_theorem_l2675_267504


namespace third_stop_off_count_l2675_267578

/-- Represents the number of people on a bus at different stops -/
structure BusOccupancy where
  initial : Nat
  after_first_stop : Nat
  after_second_stop : Nat
  after_third_stop : Nat

/-- Calculates the number of people who got off at the third stop -/
def people_off_third_stop (bus : BusOccupancy) (people_on_third : Nat) : Nat :=
  bus.after_second_stop - bus.after_third_stop + people_on_third

/-- Theorem stating the number of people who got off at the third stop -/
theorem third_stop_off_count (bus : BusOccupancy) 
  (h1 : bus.initial = 50)
  (h2 : bus.after_first_stop = bus.initial - 15)
  (h3 : bus.after_second_stop = bus.after_first_stop - 8 + 2)
  (h4 : bus.after_third_stop = 28)
  (h5 : people_on_third = 3) : 
  people_off_third_stop bus people_on_third = 4 := by
  sorry


end third_stop_off_count_l2675_267578


namespace hexagon_coloring_ways_l2675_267524

-- Define the colors
inductive Color
| Red
| Yellow
| Green

-- Define the hexagon grid
def HexagonGrid := List (List Color)

-- Define a function to check if two colors are different
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

-- Define a function to check if a coloring is valid
def valid_coloring (grid : HexagonGrid) : Prop :=
  -- Add conditions for valid coloring here
  sorry

-- Define the specific grid pattern with 8 hexagons
def specific_grid_pattern (grid : HexagonGrid) : Prop :=
  -- Add conditions for the specific grid pattern here
  sorry

-- Define the initial conditions (top-left and second-top hexagons are red)
def initial_conditions (grid : HexagonGrid) : Prop :=
  -- Add conditions for initial red hexagons here
  sorry

-- Theorem statement
theorem hexagon_coloring_ways :
  ∀ (grid : HexagonGrid),
    specific_grid_pattern grid →
    initial_conditions grid →
    valid_coloring grid →
    ∃! (n : Nat), n = 2 ∧ 
      ∃ (colorings : List HexagonGrid),
        colorings.length = n ∧
        ∀ c ∈ colorings, 
          specific_grid_pattern c ∧
          initial_conditions c ∧
          valid_coloring c :=
sorry

end hexagon_coloring_ways_l2675_267524


namespace gear_speed_proportion_l2675_267599

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  mesh_AB : A.teeth * A.speed = B.teeth * B.speed
  mesh_BC : B.teeth * B.speed = C.teeth * C.speed
  mesh_CD : C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the proportion of angular speeds for the gear system -/
theorem gear_speed_proportion (sys : GearSystem) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (sys.A.speed = k * (sys.B.teeth * sys.C.teeth * sys.D.teeth)) ∧
    (sys.B.speed = k * (sys.A.teeth * sys.C.teeth * sys.D.teeth)) ∧
    (sys.C.speed = k * (sys.A.teeth * sys.B.teeth * sys.D.teeth)) ∧
    (sys.D.speed = k * (sys.A.teeth * sys.B.teeth * sys.C.teeth)) :=
by
  sorry

end gear_speed_proportion_l2675_267599


namespace weston_penalty_kicks_l2675_267532

/-- Calculates the number of penalty kicks required for a football drill -/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: The number of penalty kicks for Weston Junior Football Club's drill is 92 -/
theorem weston_penalty_kicks :
  penalty_kicks 24 4 = 92 := by
  sorry

end weston_penalty_kicks_l2675_267532


namespace inequality_system_solutions_l2675_267556

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x + 5 > 0 ∧ x - m ≤ 1))) ↔ 
  (-3 ≤ m ∧ m < -2) :=
sorry

end inequality_system_solutions_l2675_267556


namespace prob_A_nth_day_l2675_267533

/-- The probability of switching restaurants each day -/
def switch_prob : ℝ := 0.6

/-- The probability of choosing restaurant A on the n-th day -/
def prob_A (n : ℕ) : ℝ := 0.5 + 0.5 * (-0.2)^(n - 1)

/-- Theorem stating the probability of choosing restaurant A on the n-th day -/
theorem prob_A_nth_day (n : ℕ) :
  prob_A n = 0.5 + 0.5 * (-0.2)^(n - 1) :=
by sorry

end prob_A_nth_day_l2675_267533


namespace trigonometric_problem_l2675_267589

theorem trigonometric_problem (α β : Real) 
  (h1 : 3 * Real.sin α - Real.sin β = Real.sqrt 10)
  (h2 : α + β = Real.pi / 2) :
  Real.sin α = (3 * Real.sqrt 10) / 10 ∧ 
  Real.cos (2 * β) = 4 / 5 := by
sorry

end trigonometric_problem_l2675_267589


namespace box_dimensions_l2675_267541

theorem box_dimensions (a b c : ℕ) 
  (h1 : a + c = 17) 
  (h2 : a + b = 13) 
  (h3 : b + c = 20) : 
  (a = 5 ∧ b = 8 ∧ c = 12) ∨ 
  (a = 5 ∧ b = 12 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 5 ∧ c = 12) ∨ 
  (a = 8 ∧ b = 12 ∧ c = 5) ∨ 
  (a = 12 ∧ b = 5 ∧ c = 8) ∨ 
  (a = 12 ∧ b = 8 ∧ c = 5) :=
sorry

end box_dimensions_l2675_267541


namespace max_value_theorem_l2675_267562

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a'^2 + b'^2 + c'^2 = 1 ∧
    3 * a' * b' * Real.sqrt 2 + 6 * b' * c' = 4.5 := by
  sorry

end max_value_theorem_l2675_267562


namespace president_savings_theorem_l2675_267561

/-- The amount saved by the president for his reelection campaign --/
def president_savings (total_funds friends_percentage family_percentage : ℝ) : ℝ :=
  let friends_contribution := friends_percentage * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := family_percentage * remaining_after_friends
  total_funds - friends_contribution - family_contribution

/-- Theorem stating the amount saved by the president given the campaign fund conditions --/
theorem president_savings_theorem :
  president_savings 10000 0.4 0.3 = 4200 := by
  sorry

end president_savings_theorem_l2675_267561


namespace expression_evaluation_l2675_267535

theorem expression_evaluation : (96 / 6) * 3 / 2 = 24 := by
  sorry

end expression_evaluation_l2675_267535


namespace maze_paths_count_l2675_267581

/-- Represents a branching point in the maze -/
inductive BranchingPoint
  | Major
  | Minor

/-- Represents the structure of the maze -/
structure Maze where
  entrance : Unit
  exit : Unit
  initialChoices : Nat
  majorToMinor : Nat
  minorChoices : Nat

/-- Calculates the number of paths through the maze -/
def numberOfPaths (maze : Maze) : Nat :=
  maze.initialChoices * (maze.minorChoices ^ maze.majorToMinor)

/-- The specific maze from the problem -/
def problemMaze : Maze :=
  { entrance := ()
  , exit := ()
  , initialChoices := 2
  , majorToMinor := 3
  , minorChoices := 2
  }

theorem maze_paths_count :
  numberOfPaths problemMaze = 16 := by
  sorry

#eval numberOfPaths problemMaze

end maze_paths_count_l2675_267581


namespace wholesale_price_calculation_l2675_267547

/-- Proves that the wholesale price of a machine is $90 given the specified conditions -/
theorem wholesale_price_calculation (retail_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  retail_price = 120 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (wholesale_price : ℝ),
    wholesale_price = 90 ∧
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) :=
by sorry

end wholesale_price_calculation_l2675_267547


namespace final_water_level_l2675_267574

/-- Represents the water level change in a reservoir over time -/
def waterLevelChange (initialLevel : Real) (riseRate : Real) (fallRate : Real) : Real :=
  let riseTime := 4  -- 8 a.m. to 12 p.m.
  let fallTime := 6  -- 12 p.m. to 6 p.m.
  initialLevel + riseTime * riseRate - fallTime * fallRate

/-- Theorem stating the final water level at 6 p.m. -/
theorem final_water_level (initialLevel : Real) (riseRate : Real) (fallRate : Real) :
  initialLevel = 45 ∧ riseRate = 0.6 ∧ fallRate = 0.3 →
  waterLevelChange initialLevel riseRate fallRate = 45.6 :=
by sorry

end final_water_level_l2675_267574


namespace inequality_solution_sets_l2675_267596

-- Define the inequality function
def f (m : ℝ) (x : ℝ) : Prop := m * x^2 + (2 * m - 1) * x - 2 > 0

-- Define the solution set for each case
def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then { x | x < -2 }
  else if m > 0 then { x | x < -2 ∨ x > 1/m }
  else if -1/2 < m ∧ m < 0 then { x | 1/m < x ∧ x < -2 }
  else if m = -1/2 then ∅
  else { x | -2 < x ∧ x < 1/m }

-- State the theorem
theorem inequality_solution_sets (m : ℝ) :
  { x : ℝ | f m x } = solution_set m := by sorry

end inequality_solution_sets_l2675_267596


namespace work_completion_time_l2675_267594

-- Define the work rate of A
def work_rate_A : ℚ := 1 / 60

-- Define the work done by A in 15 days
def work_done_A : ℚ := 15 * work_rate_A

-- Define the remaining work after A's 15 days
def remaining_work : ℚ := 1 - work_done_A

-- Define B's work rate based on completing the remaining work in 30 days
def work_rate_B : ℚ := remaining_work / 30

-- Define the combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Theorem to prove
theorem work_completion_time : (1 : ℚ) / combined_work_rate = 24 := by
  sorry

end work_completion_time_l2675_267594


namespace cartesian_to_polar_coords_l2675_267579

/-- Given a point P with Cartesian coordinates (1, √3), prove that its polar coordinates are (2, π/3) -/
theorem cartesian_to_polar_coords :
  let x : ℝ := 1
  let y : ℝ := Real.sqrt 3
  let ρ : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  ρ = 2 ∧ θ = π / 3 := by sorry

end cartesian_to_polar_coords_l2675_267579


namespace missing_digit_divisible_by_three_l2675_267588

theorem missing_digit_divisible_by_three :
  ∃ d : ℕ, d < 10 ∧ (43500 + d * 10 + 1) % 3 = 0 :=
by
  sorry

end missing_digit_divisible_by_three_l2675_267588


namespace smallest_a_parabola_l2675_267566

/-- The smallest possible value of 'a' for a parabola with specific conditions -/
theorem smallest_a_parabola : 
  ∀ (a b c : ℝ), 
  (∃ (x y : ℝ), y = a * x^2 + b * x + c ∧ x = 3/2 ∧ y = -1/4) →  -- vertex condition
  (a > 0) →  -- a is positive
  (∃ (n : ℤ), 2*a + b + 3*c = n) →  -- 2a + b + 3c is an integer
  (∀ (a' : ℝ), 
    (∃ (b' c' : ℝ), 
      (∃ (x y : ℝ), y = a' * x^2 + b' * x + c' ∧ x = 3/2 ∧ y = -1/4) ∧
      (a' > 0) ∧
      (∃ (n : ℤ), 2*a' + b' + 3*c' = n)) → 
    a ≤ a') →
  a = 3/23 := by
sorry

end smallest_a_parabola_l2675_267566


namespace least_positive_angle_theta_l2675_267572

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0) → 
  (∀ φ, φ > 0 → φ < θ → Real.cos (15 * Real.pi / 180) ≠ Real.sin (35 * Real.pi / 180) + Real.sin φ) → 
  Real.cos (15 * Real.pi / 180) = Real.sin (35 * Real.pi / 180) + Real.sin θ → 
  θ = 55 * Real.pi / 180 := by
sorry

end least_positive_angle_theta_l2675_267572


namespace painting_fraction_l2675_267592

def total_students : ℕ := 50
def field_fraction : ℚ := 1 / 5
def classroom_left : ℕ := 10

theorem painting_fraction :
  (total_students - (field_fraction * total_students).num - classroom_left) / total_students = 3 / 5 := by
  sorry

end painting_fraction_l2675_267592


namespace carlos_blocks_l2675_267501

theorem carlos_blocks (initial_blocks : ℕ) (given_blocks : ℕ) : 
  initial_blocks = 58 → given_blocks = 21 → initial_blocks - given_blocks = 37 := by
  sorry

end carlos_blocks_l2675_267501


namespace range_of_z_l2675_267507

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  4 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 12 := by
  sorry

end range_of_z_l2675_267507


namespace max_square_plots_l2675_267558

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing -/
def availableFencing : ℕ := 2500

/-- Calculates the number of square plots given the side length -/
def numPlots (fd : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (fd.width / sideLength) * (fd.length / sideLength)

/-- Calculates the required internal fencing given the side length -/
def requiredFencing (fd : FieldDimensions) (sideLength : ℕ) : ℕ :=
  fd.width * ((fd.length / sideLength) - 1) + fd.length * ((fd.width / sideLength) - 1)

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (fd : FieldDimensions) 
    (h1 : fd.width = 30) 
    (h2 : fd.length = 60) : 
    ∃ (sideLength : ℕ), 
      sideLength > 0 ∧ 
      fd.width % sideLength = 0 ∧ 
      fd.length % sideLength = 0 ∧
      requiredFencing fd sideLength ≤ availableFencing ∧
      ∀ (s : ℕ), s > 0 → 
        fd.width % s = 0 → 
        fd.length % s = 0 → 
        requiredFencing fd s ≤ availableFencing → 
        numPlots fd s ≤ numPlots fd sideLength :=
  sorry

#eval numPlots ⟨30, 60⟩ 5  -- Should evaluate to 72

end max_square_plots_l2675_267558


namespace infinite_divisibility_l2675_267516

theorem infinite_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_mod : p % 4 = 1) (h_not_17 : p ≠ 17) :
  let n := p
  ∃ k : Nat, 3^((n - 2)^(n - 1) - 1) - 1 = 17 * n^2 * k := by
  sorry

end infinite_divisibility_l2675_267516


namespace M_properly_contains_N_l2675_267575

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}
def N : Set ℝ := {x : ℝ | ∃ y, y = Real.log (x - 2)}

-- Theorem stating that M properly contains N
theorem M_properly_contains_N : M ⊃ N := by
  sorry

end M_properly_contains_N_l2675_267575


namespace min_k_for_f_geq_3_solution_set_f_lt_3x_l2675_267557

-- Define the function f(x, k)
def f (x k : ℝ) : ℝ := |x - 3| + |x - 2| + k

-- Theorem for part I
theorem min_k_for_f_geq_3 :
  (∀ x : ℝ, f x 2 ≥ 3) ∧ (∀ k < 2, ∃ x : ℝ, f x k < 3) :=
sorry

-- Theorem for part II
theorem solution_set_f_lt_3x :
  {x : ℝ | f x 1 < 3 * x} = {x : ℝ | x > 6/5} :=
sorry

end min_k_for_f_geq_3_solution_set_f_lt_3x_l2675_267557
