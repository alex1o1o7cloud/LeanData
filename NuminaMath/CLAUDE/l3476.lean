import Mathlib

namespace negation_equivalence_l3476_347632

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, (x₀ + 1 < 0) ∨ (x₀^2 - x₀ > 0)) ↔ 
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0)) :=
by sorry

end negation_equivalence_l3476_347632


namespace fraction_equality_l3476_347679

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 4)
  (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 := by
sorry

end fraction_equality_l3476_347679


namespace amy_bob_games_l3476_347657

/-- Represents the total number of players -/
def total_players : ℕ := 12

/-- Represents the number of players in each game -/
def players_per_game : ℕ := 6

/-- Represents the number of players that are always together (Chris and Dave) -/
def always_together : ℕ := 2

/-- Represents the number of specific players we're interested in (Amy and Bob) -/
def specific_players : ℕ := 2

/-- Theorem stating that the number of games where Amy and Bob play together
    is equal to the number of ways to choose 2 players from the remaining 8 players -/
theorem amy_bob_games :
  (total_players - specific_players - always_together).choose 2 =
  Nat.choose 8 2 := by sorry

end amy_bob_games_l3476_347657


namespace inscribed_rectangle_area_l3476_347601

/-- A right triangle with an inscribed rectangle -/
structure InscribedRectangle where
  /-- Length of AB in the right triangle AGD -/
  ab : ℝ
  /-- Length of CD in the right triangle AGD -/
  cd : ℝ
  /-- Length of BC in the inscribed rectangle BCFE -/
  bc : ℝ
  /-- Length of FE in the inscribed rectangle BCFE -/
  fe : ℝ
  /-- BC is parallel to AD -/
  bc_parallel_ad : True
  /-- FE is parallel to AD -/
  fe_parallel_ad : True
  /-- Length of BC is one-third of FE -/
  bc_one_third_fe : bc = fe / 3
  /-- AB = 40 units -/
  ab_eq_40 : ab = 40
  /-- CD = 70 units -/
  cd_eq_70 : cd = 70

/-- The area of the inscribed rectangle BCFE is 2800 square units -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) : 
  rect.bc * rect.fe = 2800 := by
  sorry

end inscribed_rectangle_area_l3476_347601


namespace print_shop_charge_difference_l3476_347617

/-- The cost per color copy at print shop X -/
def cost_x : ℚ := 1.20

/-- The cost per color copy at print shop Y -/
def cost_y : ℚ := 1.70

/-- The number of color copies -/
def num_copies : ℕ := 70

/-- The difference in charge between print shop Y and print shop X for a given number of copies -/
def charge_difference (n : ℕ) : ℚ := n * (cost_y - cost_x)

theorem print_shop_charge_difference : 
  charge_difference num_copies = 35 := by sorry

end print_shop_charge_difference_l3476_347617


namespace f_continuous_not_bounded_variation_l3476_347648

/-- The function f(x) = x sin(1/x) for x ≠ 0 and f(0) = 0 -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x * Real.sin (1 / x) else 0

/-- The interval [0, 1] -/
def I : Set ℝ := Set.Icc 0 1

theorem f_continuous_not_bounded_variation :
  ContinuousOn f I ∧ ¬ BoundedVariationOn f I := by sorry

end f_continuous_not_bounded_variation_l3476_347648


namespace disease_cases_1975_l3476_347603

/-- Calculates the number of disease cases in a given year, assuming a linear decrease -/
def diseaseCases (initialYear finalYear : ℕ) (initialCases finalCases : ℕ) (targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let annualDecrease := totalDecrease / totalYears
  let yearsPassed := targetYear - initialYear
  initialCases - (annualDecrease * yearsPassed)

theorem disease_cases_1975 :
  diseaseCases 1950 2000 500000 1000 1975 = 250500 := by
  sorry

end disease_cases_1975_l3476_347603


namespace part1_part2_l3476_347600

-- Define complex numbers
def z1 (m : ℝ) : ℂ := m - 2*Complex.I
def z2 (n : ℝ) : ℂ := 1 - n*Complex.I

-- Part 1
theorem part1 : Complex.abs (z1 1 + z2 (-1)) = Real.sqrt 5 := by sorry

-- Part 2
theorem part2 : z1 0 = (z2 1)^2 := by sorry

end part1_part2_l3476_347600


namespace two_negative_solutions_iff_b_in_range_l3476_347602

/-- The equation 9^x + |3^x + b| = 5 has exactly two negative real number solutions if and only if b is in the open interval (-5.25, -5) -/
theorem two_negative_solutions_iff_b_in_range (b : ℝ) : 
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
    (9^x + |3^x + b| = 5) ∧ 
    (9^y + |3^y + b| = 5) ∧
    (∀ z : ℝ, z < 0 → z ≠ x → z ≠ y → 9^z + |3^z + b| ≠ 5)) ↔ 
  -5.25 < b ∧ b < -5 :=
sorry

end two_negative_solutions_iff_b_in_range_l3476_347602


namespace product_profit_l3476_347614

theorem product_profit (original_price : ℝ) (cost_price : ℝ) : 
  cost_price > 0 →
  original_price > 0 →
  (0.8 * original_price = 1.2 * cost_price) →
  (original_price - cost_price) / cost_price = 0.5 := by
sorry

end product_profit_l3476_347614


namespace median_and_mode_are_23_l3476_347645

/-- Represents the shoe size distribution of a class --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of a shoe size distribution --/
def median (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- The shoe size distribution for the given class --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40 }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 := by
  sorry

end median_and_mode_are_23_l3476_347645


namespace four_digit_int_solution_l3476_347635

/-- Represents a four-digit positive integer -/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : 0 < a
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

/-- Converts a FourDigitInt to a natural number -/
def FourDigitInt.toNat (n : FourDigitInt) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem four_digit_int_solution :
  ∃! (n : FourDigitInt),
    n.a + n.b + n.c + n.d = 16 ∧
    n.b + n.c = 10 ∧
    n.a - n.d = 2 ∧
    n.toNat % 11 = 0 ∧
    n.toNat = 4642 := by
  sorry

#check four_digit_int_solution

end four_digit_int_solution_l3476_347635


namespace no_self_power_divisibility_l3476_347675

theorem no_self_power_divisibility (n : ℕ) : n > 1 → ¬(n ∣ 2^n - 1) := by
  sorry

end no_self_power_divisibility_l3476_347675


namespace annika_hiking_time_l3476_347647

/-- Annika's hiking problem -/
theorem annika_hiking_time (rate : ℝ) (initial_distance : ℝ) (total_distance : ℝ) : 
  rate = 12 →
  initial_distance = 2.75 →
  total_distance = 3.5 →
  (total_distance - initial_distance) * rate + total_distance * rate = 51 := by
sorry

end annika_hiking_time_l3476_347647


namespace greatest_integer_with_gcd_six_one_thirty_eight_satisfies_conditions_one_thirty_eight_is_greatest_main_result_l3476_347661

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem one_thirty_eight_satisfies_conditions : 138 < 150 ∧ Nat.gcd 138 18 = 6 :=
by sorry

theorem one_thirty_eight_is_greatest : 
  ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ 138 :=
by sorry

theorem main_result : 
  (∃ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ 
    ∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n) ∧
  (∀ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ 
    (∀ m : ℕ, m < 150 ∧ Nat.gcd m 18 = 6 → m ≤ n) → n = 138) :=
by sorry

end greatest_integer_with_gcd_six_one_thirty_eight_satisfies_conditions_one_thirty_eight_is_greatest_main_result_l3476_347661


namespace function_properties_l3476_347686

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x + b

theorem function_properties (a b : ℝ) :
  (∀ x y : ℝ, x = 1 → y = f a b x → (3 * x - y - 3 = 0) → (a = -2 ∧ b = -1/2)) ∧
  ((∀ x : ℝ, x ≠ 0 → (deriv (f a b) x = 0 ↔ x = 1)) → a = 1) ∧
  ((-2 ≤ a ∧ a < 0) →
    (∃ m : ℝ, m = 12 ∧
      (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ ≤ 2 ∧ 0 < x₂ ∧ x₂ ≤ 2 →
        |f a b x₁ - f a b x₂| ≤ m * |1/x₁ - 1/x₂|) ∧
      (∀ m' : ℝ, m' < m →
        ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ ≤ 2 ∧ 0 < x₂ ∧ x₂ ≤ 2 ∧
          |f a b x₁ - f a b x₂| > m' * |1/x₁ - 1/x₂|))) :=
by sorry

end function_properties_l3476_347686


namespace yellow_marbles_count_l3476_347668

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by sorry

end yellow_marbles_count_l3476_347668


namespace math_correct_percentage_l3476_347669

/-- Represents the number of questions in the math test -/
def math_questions : ℕ := 40

/-- Represents the number of questions in the English test -/
def english_questions : ℕ := 50

/-- Represents the percentage of English questions answered correctly -/
def english_correct_percentage : ℚ := 98 / 100

/-- Represents the total number of questions answered correctly across both tests -/
def total_correct : ℕ := 79

/-- Theorem stating that the percentage of math questions answered correctly is 75% -/
theorem math_correct_percentage :
  (total_correct - (english_correct_percentage * english_questions).num) / math_questions = 75 / 100 := by
  sorry

end math_correct_percentage_l3476_347669


namespace problem_solution_l3476_347628

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := by
  sorry

end problem_solution_l3476_347628


namespace sum_vertices_is_nine_l3476_347682

/-- The number of vertices in a rectangle -/
def rectangle_vertices : ℕ := 4

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- The sum of vertices of a rectangle and a pentagon -/
def sum_vertices : ℕ := rectangle_vertices + pentagon_vertices

theorem sum_vertices_is_nine : sum_vertices = 9 := by
  sorry

end sum_vertices_is_nine_l3476_347682


namespace sqrt_equality_implies_inequality_l3476_347620

theorem sqrt_equality_implies_inequality (b : ℝ) : 
  Real.sqrt ((3 - b)^2) = 3 - b → b ≤ 3 := by
sorry

end sqrt_equality_implies_inequality_l3476_347620


namespace james_two_semester_cost_l3476_347693

/-- The cost of James's two semesters at community college -/
def two_semester_cost (units_per_semester : ℕ) (cost_per_unit : ℕ) : ℕ :=
  2 * units_per_semester * cost_per_unit

/-- Proof that James pays $2000 for two semesters -/
theorem james_two_semester_cost :
  two_semester_cost 20 50 = 2000 := by
  sorry

#eval two_semester_cost 20 50

end james_two_semester_cost_l3476_347693


namespace range_of_f_l3476_347633

def f (x : ℝ) := x^2 + 4*x + 6

theorem range_of_f : 
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Ico (-3) 0, f x = y ∧
  ∀ x ∈ Set.Ico (-3) 0, 2 ≤ f x ∧ f x < 6 :=
sorry

end range_of_f_l3476_347633


namespace inequality_not_always_true_l3476_347650

theorem inequality_not_always_true (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬ (∀ b, c * b^2 < a * b^2) :=
sorry

end inequality_not_always_true_l3476_347650


namespace laundry_dry_cycle_time_l3476_347660

theorem laundry_dry_cycle_time 
  (total_loads : ℕ) 
  (wash_time_per_load : ℚ) 
  (total_time : ℚ) 
  (h1 : total_loads = 8) 
  (h2 : wash_time_per_load = 45 / 60) 
  (h3 : total_time = 14) : 
  (total_time - (total_loads : ℚ) * wash_time_per_load) / total_loads = 1 := by
  sorry

end laundry_dry_cycle_time_l3476_347660


namespace fraction_multiplication_l3476_347659

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 5 / 9 * 11 / 13 = 440 / 2457 := by
  sorry

end fraction_multiplication_l3476_347659


namespace range_of_a_l3476_347667

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 1 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, Real.exp (2*x) - 2 * Real.exp x + a ≥ 0

-- Define the theorem
theorem range_of_a : 
  ∀ a : ℝ, (p a ∧ q a) → a ∈ Set.Ici 2 := by sorry

end range_of_a_l3476_347667


namespace option_A_correct_option_C_correct_l3476_347664

-- Define the set M
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Define the set B
def B : Set ℤ := {b | ∃ n : ℕ, b = 2*n + 1}

-- Theorem for option A
theorem option_A_correct : ∀ a₁ a₂ : ℤ, a₁ ∈ M → a₂ ∈ M → (a₁ * a₂) ∈ M := by
  sorry

-- Theorem for option C
theorem option_C_correct : B ⊆ M := by
  sorry

end option_A_correct_option_C_correct_l3476_347664


namespace trig_identity_l3476_347674

-- Statement of the trigonometric identity
theorem trig_identity (α β γ : ℝ) :
  Real.sin α + Real.sin β + Real.sin γ = 4 * Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) := by
  sorry

end trig_identity_l3476_347674


namespace catrionas_fish_count_catrionas_aquarium_l3476_347624

theorem catrionas_fish_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun goldfish angelfish guppies total =>
    (goldfish = 8) →
    (angelfish = goldfish + 4) →
    (guppies = 2 * angelfish) →
    (total = goldfish + angelfish + guppies) →
    (total = 44)

-- Proof
theorem catrionas_aquarium : catrionas_fish_count 8 12 24 44 := by
  sorry

end catrionas_fish_count_catrionas_aquarium_l3476_347624


namespace complex_equation_solution_l3476_347636

theorem complex_equation_solution (z : ℂ) : 2 + z = (2 - z) * Complex.I → z = 2 * Complex.I := by
  sorry

end complex_equation_solution_l3476_347636


namespace permutation_equation_solution_l3476_347666

theorem permutation_equation_solution (m : ℕ) : 
  (m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) → m = 5 :=
by sorry

end permutation_equation_solution_l3476_347666


namespace area_outside_parallel_chords_l3476_347696

/-- Given a circle with radius 10 inches and two equal parallel chords 10 inches apart,
    the area of the region outside these chords but inside the circle is (200π/3 - 25√3) square inches. -/
theorem area_outside_parallel_chords (r : ℝ) (d : ℝ) : 
  r = 10 → d = 10 → 
  (2 * π * r^2 / 3 - 5 * r * Real.sqrt 3) = (200 * π / 3 - 25 * Real.sqrt 3) :=
by sorry

end area_outside_parallel_chords_l3476_347696


namespace homework_problem_l3476_347619

theorem homework_problem (total : ℕ) (finished_ratio unfinished_ratio : ℕ) 
  (h_total : total = 65)
  (h_ratio : finished_ratio = 9 ∧ unfinished_ratio = 4) :
  (finished_ratio * total) / (finished_ratio + unfinished_ratio) = 45 := by
  sorry

end homework_problem_l3476_347619


namespace not_divisible_by_nine_l3476_347677

theorem not_divisible_by_nine (t : ℤ) (k : ℤ) (h : k = 9 * t + 8) :
  ¬ (9 ∣ (5 * (9 * t + 8) * (9 * 25 * t + 222))) := by
  sorry

end not_divisible_by_nine_l3476_347677


namespace trigonometric_problem_l3476_347608

theorem trigonometric_problem (α β : Real) 
  (h1 : 2 * Real.sin α = 2 * Real.sin (α / 2) ^ 2 - 1)
  (h2 : α ∈ Set.Ioo 0 Real.pi)
  (h3 : β ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h4 : 3 * Real.tan β ^ 2 - 2 * Real.tan β = 1) :
  (Real.sin (2 * α) + Real.cos (2 * α) = -1/5) ∧ 
  (α + β = 7 * Real.pi / 4) := by
  sorry

end trigonometric_problem_l3476_347608


namespace coat_cost_l3476_347627

def weekly_savings : ℕ := 25
def weeks_of_saving : ℕ := 6
def bill_fraction : ℚ := 1 / 3
def dad_contribution : ℕ := 70

theorem coat_cost : 
  weekly_savings * weeks_of_saving - 
  (weekly_savings * weeks_of_saving : ℚ) * bill_fraction +
  dad_contribution = 170 := by sorry

end coat_cost_l3476_347627


namespace system_of_equations_solution_l3476_347609

theorem system_of_equations_solution :
  ∃! (x y : ℚ), (5 * x - 3 * y = -7) ∧ (2 * x + 7 * y = -26) ∧ 
  (x = -127 / 41) ∧ (y = -116 / 41) := by
  sorry

end system_of_equations_solution_l3476_347609


namespace prime_divides_square_minus_prime_l3476_347611

theorem prime_divides_square_minus_prime (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (q : ℕ) (n : ℕ+), q.Prime ∧ q < p ∧ p ∣ n.val^2 - q := by
  sorry

end prime_divides_square_minus_prime_l3476_347611


namespace new_crew_weight_l3476_347676

/-- The combined weight of two new crew members in a sailboat scenario -/
theorem new_crew_weight (n : ℕ) (avg_increase w1 w2 : ℝ) : 
  n = 12 → 
  avg_increase = 2.2 →
  w1 = 78 →
  w2 = 65 →
  (n : ℝ) * avg_increase + w1 + w2 = 169.4 :=
by sorry

end new_crew_weight_l3476_347676


namespace complete_square_quadratic_l3476_347687

theorem complete_square_quadratic (x : ℝ) : 
  x^2 - 2*x - 2 = 0 → (x - 1)^2 = 3 := by
  sorry

end complete_square_quadratic_l3476_347687


namespace triangle_square_perimeter_l3476_347640

theorem triangle_square_perimeter (d : ℕ) : 
  (∃ (s t : ℝ), 
    s > 0 ∧ 
    3 * t - 4 * s = 2016 ∧ 
    t - s = d) ↔ 
  d > 672 :=
sorry

end triangle_square_perimeter_l3476_347640


namespace max_sum_abc_l3476_347695

/-- Definition of A_n -/
def A_n (a n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9

/-- Definition of B_n -/
def B_n (b n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9

/-- Definition of C_n -/
def C_n (c n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

/-- The main theorem -/
theorem max_sum_abc :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧
                 1 ≤ b ∧ b ≤ 9 ∧
                 1 ≤ c ∧ c ≤ 9 ∧
                 (∃ (n : ℕ), n > 0 ∧ C_n c n - B_n b n = (A_n a n)^2) ∧
                 a + b + c = 18 ∧
                 ∀ (a' b' c' : ℕ), 1 ≤ a' ∧ a' ≤ 9 ∧
                                   1 ≤ b' ∧ b' ≤ 9 ∧
                                   1 ≤ c' ∧ c' ≤ 9 ∧
                                   (∃ (n : ℕ), n > 0 ∧ C_n c' n - B_n b' n = (A_n a' n)^2) →
                                   a' + b' + c' ≤ 18 :=
by sorry

end max_sum_abc_l3476_347695


namespace symmetric_point_coordinates_l3476_347690

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero. -/
def symmetricToOrigin (p q : Point) : Prop :=
  p.x + q.x = 0 ∧ p.y + q.y = 0

theorem symmetric_point_coordinates :
  let A : Point := ⟨2, 4⟩
  let B : Point := ⟨-2, -4⟩
  symmetricToOrigin A B → B = ⟨-2, -4⟩ := by
  sorry

end symmetric_point_coordinates_l3476_347690


namespace sqrt_50_plus_sqrt_32_l3476_347683

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end sqrt_50_plus_sqrt_32_l3476_347683


namespace compute_expression_l3476_347637

theorem compute_expression : (75 * 2424 + 25 * 2424) / 2 = 121200 := by
  sorry

end compute_expression_l3476_347637


namespace social_gathering_handshakes_l3476_347616

/-- Represents a social gathering with specific conditions -/
structure SocialGathering where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  group_b_connected : Nat
  group_b_isolated : Nat
  a_to_b_connections : Nat
  a_to_b_per_person : Nat

/-- Calculates the number of handshakes in the social gathering -/
def count_handshakes (g : SocialGathering) : Nat :=
  let a_to_b_handshakes := (g.group_a_size - g.a_to_b_connections) * g.group_b_size
  let b_connected_handshakes := g.group_b_connected * (g.group_b_connected - 1) / 2
  let b_isolated_handshakes := g.group_b_isolated * (g.group_b_isolated - 1) / 2 + g.group_b_isolated * g.group_b_connected
  a_to_b_handshakes + b_connected_handshakes + b_isolated_handshakes

/-- The main theorem stating the number of handshakes in the given social gathering -/
theorem social_gathering_handshakes :
  let g : SocialGathering := {
    total_people := 30,
    group_a_size := 15,
    group_b_size := 15,
    group_b_connected := 10,
    group_b_isolated := 5,
    a_to_b_connections := 5,
    a_to_b_per_person := 3
  }
  count_handshakes g = 255 := by
  sorry


end social_gathering_handshakes_l3476_347616


namespace student_average_greater_than_true_average_l3476_347656

theorem student_average_greater_than_true_average 
  (u v w x y : ℝ) 
  (h : u ≤ v ∧ v ≤ w ∧ w ≤ x ∧ x ≤ y) : 
  ((u + v + w) / 3 + x + y) / 3 > (u + v + w + x + y) / 5 := by
  sorry

end student_average_greater_than_true_average_l3476_347656


namespace platform_length_l3476_347641

/-- Calculates the length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmph = 54 →
  crossing_time = 25 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * crossing_time
  let platform_length := total_distance - train_length
  platform_length = 175 := by sorry

end platform_length_l3476_347641


namespace collinear_points_theorem_l3476_347662

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if points A(a, 1), B(9, 0), and C(-3, 4) are collinear, then a = 6. -/
theorem collinear_points_theorem (a : ℝ) :
  collinear a 1 9 0 (-3) 4 → a = 6 := by
  sorry

#check collinear_points_theorem

end collinear_points_theorem_l3476_347662


namespace alyssa_grew_nine_turnips_l3476_347658

/-- The number of turnips Keith grew -/
def keith_turnips : ℕ := 6

/-- The total number of turnips Keith and Alyssa grew together -/
def total_turnips : ℕ := 15

/-- The number of turnips Alyssa grew -/
def alyssa_turnips : ℕ := total_turnips - keith_turnips

theorem alyssa_grew_nine_turnips : alyssa_turnips = 9 := by
  sorry

end alyssa_grew_nine_turnips_l3476_347658


namespace parallel_lines_angle_problem_l3476_347644

-- Define the angles as real numbers
variable (AXE CYX BXY : ℝ)

-- State the theorem
theorem parallel_lines_angle_problem 
  (h1 : AXE = 4 * CYX - 120) -- Given condition
  (h2 : AXE = CYX) -- From parallel lines property
  : BXY = 40 := by
  sorry

end parallel_lines_angle_problem_l3476_347644


namespace parabola_from_hyperbola_l3476_347654

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  (positive_p : 0 < p)

/-- The center of a hyperbola -/
def Hyperbola.center (h : Hyperbola) : ℝ × ℝ := (0, 0)

/-- The right focus of a hyperbola -/
def Hyperbola.right_focus (h : Hyperbola) : ℝ × ℝ := (h.a, 0)

/-- The equation of a parabola with vertex at the origin -/
def Parabola.equation (p : Parabola) (x y : ℝ) : Prop :=
  y^2 = 4 * p.p * x

theorem parabola_from_hyperbola (h : Hyperbola) 
    (h_eq : ∀ x y : ℝ, x^2 / 4 - y^2 / 5 = 1 ↔ (x / h.a)^2 - (y / h.b)^2 = 1) :
    ∃ p : Parabola, 
      p.equation = fun x y => y^2 = 12 * x ∧
      Parabola.equation p x y ↔ y^2 = 12 * x := by
  sorry

end parabola_from_hyperbola_l3476_347654


namespace rectangle_triangle_ef_length_l3476_347634

/-- Given a rectangle ABCD with side lengths AB and BC, and a triangle DEF inside it
    where DE = DF and the area of DEF is one-third of ABCD's area,
    prove that EF has length 12 when AB = 9 and BC = 12. -/
theorem rectangle_triangle_ef_length
  (AB BC : ℝ)
  (DE DF EF : ℝ)
  (h_ab : AB = 9)
  (h_bc : BC = 12)
  (h_de_df : DE = DF)
  (h_area : (1/2) * DE * DF = (1/3) * AB * BC) :
  EF = 12 := by
  sorry

end rectangle_triangle_ef_length_l3476_347634


namespace opposite_of_2023_l3476_347631

theorem opposite_of_2023 : -(2023 : ℝ) = -2023 := by sorry

end opposite_of_2023_l3476_347631


namespace even_implies_symmetric_at_most_one_intersection_l3476_347665

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define symmetry about a point
def symmetric_about (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Theorem 1: If f(x+1) is even, then f(x) is symmetric about x = 1
theorem even_implies_symmetric :
  is_even (fun x ↦ f (x + 1)) → symmetric_about f 1 := by sorry

-- Theorem 2: Any function has at most one intersection with a vertical line
theorem at_most_one_intersection (a : ℝ) :
  ∃! y, f y = a := by sorry

end even_implies_symmetric_at_most_one_intersection_l3476_347665


namespace number_problem_l3476_347610

theorem number_problem (x n : ℝ) : x = 4 ∧ n * x + 3 = 10 * x - 17 → n = 5 := by
  sorry

end number_problem_l3476_347610


namespace square_root_property_l3476_347629

theorem square_root_property (x : ℝ) : 
  (Real.sqrt (2*x + 3) = 3) → (2*x + 3)^2 = 81 := by
  sorry

end square_root_property_l3476_347629


namespace sum_of_five_integers_l3476_347639

theorem sum_of_five_integers (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12 →
  a + b + c + d + e = 17 :=
by sorry

end sum_of_five_integers_l3476_347639


namespace even_a_iff_xor_inequality_l3476_347689

-- Define bitwise XOR operation
def bitwiseXOR (a b : ℕ) : ℕ :=
  Nat.rec 0 (fun k res => 
    if (a / 2^k + b / 2^k - res / 2^k) % 2 = 0 
    then res 
    else res + 2^k) a

-- State the theorem
theorem even_a_iff_xor_inequality (a : ℕ) : 
  (a > 0 ∧ a % 2 = 0) ↔ 
  (∀ x y : ℕ, x > y → bitwiseXOR x (a * x) ≠ bitwiseXOR y (a * y)) := by
sorry

end even_a_iff_xor_inequality_l3476_347689


namespace common_volume_for_ratios_l3476_347613

/-- The volume of the common part of two identical triangular pyramids -/
noncomputable def common_volume (V : ℝ) (r : ℝ) : ℝ := sorry

/-- Theorem stating the volume of the common part for different ratios -/
theorem common_volume_for_ratios (V : ℝ) (V_pos : V > 0) :
  (common_volume V (1/2) = 2/3 * V) ∧
  (common_volume V (3/4) = 1/2 * V) ∧
  (common_volume V (2/3) = 110/243 * V) ∧
  (common_volume V (4/5) = 12/25 * V) := by sorry

end common_volume_for_ratios_l3476_347613


namespace middle_number_is_four_l3476_347692

def is_valid_triple (a b c : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ a + b + c = 13

def multiple_possibilities_for_bc (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ is_valid_triple a b₁ c₁ ∧ is_valid_triple a b₂ c₂

def multiple_possibilities_for_ab (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ is_valid_triple a₁ b₁ c ∧ is_valid_triple a₂ b₂ c

def multiple_possibilities_for_ac (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, a₁ ≠ a₂ ∧ is_valid_triple a₁ b c₁ ∧ is_valid_triple a₂ b c₂

theorem middle_number_is_four (a b c : ℕ) :
  is_valid_triple a b c →
  multiple_possibilities_for_bc a →
  multiple_possibilities_for_ab c →
  multiple_possibilities_for_ac b →
  b = 4 := by
  sorry

end middle_number_is_four_l3476_347692


namespace divide_by_four_theorem_l3476_347685

theorem divide_by_four_theorem (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 := by
  sorry

end divide_by_four_theorem_l3476_347685


namespace fibonacci_ratio_property_fibonacci_ratio_periodic_fibonacci_ratio_distinct_in_period_l3476_347615

/-- Fibonacci sequence -/
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

/-- Ratio of consecutive Fibonacci numbers -/
def fibonacciRatio (n : ℕ) : ℚ :=
  if n = 0 then 0 else (fibonacci (n + 1) : ℚ) / (fibonacci n : ℚ)

theorem fibonacci_ratio_property (n : ℕ) (h : n > 1) :
  fibonacciRatio n = 1 + 1 / (fibonacciRatio (n - 1)) :=
sorry

theorem fibonacci_ratio_periodic :
  ∃ (p : ℕ) (h : p > 0), ∀ (n : ℕ), fibonacciRatio (n + p) = fibonacciRatio n :=
sorry

theorem fibonacci_ratio_distinct_in_period (p : ℕ) (h : p > 0) :
  ∀ (i j : ℕ), i < j → j < p → fibonacciRatio i ≠ fibonacciRatio j :=
sorry

end fibonacci_ratio_property_fibonacci_ratio_periodic_fibonacci_ratio_distinct_in_period_l3476_347615


namespace floor_plus_x_equation_l3476_347638

theorem floor_plus_x_equation :
  ∃! x : ℝ, ⌊x⌋ + x = 20.5 :=
by
  sorry

end floor_plus_x_equation_l3476_347638


namespace series_sum_l3476_347651

open Real

/-- The sum of the series ∑_{n=1}^{∞} (sin^n x) / n for x ≠ π/2 + 2πk, where k is an integer -/
theorem series_sum (x : ℝ) (h : ∀ k : ℤ, x ≠ π / 2 + 2 * π * k) :
  ∑' n, (sin x) ^ n / n = -log (1 - sin x) :=
by sorry


end series_sum_l3476_347651


namespace log_half_increasing_interval_l3476_347607

noncomputable def y (x a : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem log_half_increasing_interval (a : ℝ) (h : a > 0) :
  (∃ m, m > 0 ∧ is_increasing (y · a) 0 m) ↔
  ((0 < a ∧ a ≤ Real.sqrt 3 ∧ ∃ m, 0 < m ∧ m ≤ a) ∨
   (a > Real.sqrt 3 ∧ ∃ m, 0 < m ∧ m ≤ a - Real.sqrt (a^2 - 3))) :=
sorry

end log_half_increasing_interval_l3476_347607


namespace helium_pressure_change_l3476_347670

/-- Boyle's law for ideal gases at constant temperature -/
axiom boyles_law {p1 p2 v1 v2 : ℝ} (h : p1 * v1 = p2 * v2) : 
  p1 * v1 = p2 * v2

theorem helium_pressure_change (p1 v1 p2 v2 : ℝ) 
  (h1 : p1 = 4) 
  (h2 : v1 = 3) 
  (h3 : v2 = 6) 
  (h4 : p1 * v1 = p2 * v2) : 
  p2 = 2 := by
  sorry

#check helium_pressure_change

end helium_pressure_change_l3476_347670


namespace board_number_after_hour_l3476_347681

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def next_number (n : ℕ) : ℕ :=
  digit_product n + 15

def iterate_operation (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterate_operation (next_number n) k

theorem board_number_after_hour (initial : ℕ) (h : initial = 98) :
  iterate_operation initial 60 = 24 :=
sorry

end board_number_after_hour_l3476_347681


namespace division_problem_l3476_347672

theorem division_problem :
  let dividend : Nat := 73648
  let divisor : Nat := 874
  let quotient : Nat := dividend / divisor
  let remainder : Nat := dividend % divisor
  (quotient = 84) ∧ 
  (remainder = 232) ∧ 
  (remainder + 375 = 607) := by
  sorry

end division_problem_l3476_347672


namespace total_dice_count_l3476_347699

theorem total_dice_count (ivan_dice : ℕ) (jerry_dice : ℕ) : 
  ivan_dice = 20 → 
  jerry_dice = 2 * ivan_dice → 
  ivan_dice + jerry_dice = 60 := by
  sorry

end total_dice_count_l3476_347699


namespace minimum_point_l3476_347652

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 4| - 2

-- State the theorem
theorem minimum_point :
  ∃! p : ℝ × ℝ, p.1 = 4 ∧ p.2 = -2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end minimum_point_l3476_347652


namespace class_size_problem_l3476_347655

theorem class_size_problem (passing_score : ℝ) (class_average : ℝ) 
  (pass_average_before : ℝ) (fail_average_before : ℝ)
  (pass_average_after : ℝ) (fail_average_after : ℝ)
  (points_added : ℝ) :
  passing_score = 65 →
  class_average = 66 →
  pass_average_before = 71 →
  fail_average_before = 56 →
  pass_average_after = 75 →
  fail_average_after = 59 →
  points_added = 5 →
  ∃ (total_students : ℕ), 
    15 < total_students ∧ 
    total_students < 30 ∧
    total_students = 24 :=
by sorry

end class_size_problem_l3476_347655


namespace line_points_k_value_l3476_347684

/-- A line contains the points (4, 10), (-4, k), and (-12, 6). Prove that k = 8. -/
theorem line_points_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (10 = m * 4 + b) ∧ 
    (k = m * (-4) + b) ∧ 
    (6 = m * (-12) + b)) → 
  k = 8 := by
sorry

end line_points_k_value_l3476_347684


namespace max_m_value_l3476_347698

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m * a * b / (3 * a + b) ≤ a + 3 * b) ↔ m ≤ 16 :=
sorry

end max_m_value_l3476_347698


namespace union_of_A_and_B_l3476_347694

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 = 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 1 = 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 1, 5} := by
  sorry

end union_of_A_and_B_l3476_347694


namespace pairing_fraction_l3476_347623

theorem pairing_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) :
  n = 4 * s / 3 →
  (s / 3 + n / 4) / (s + n) = 2 / 7 := by
sorry

end pairing_fraction_l3476_347623


namespace floor_difference_equals_ten_l3476_347663

theorem floor_difference_equals_ten : 
  ⌊(2010^4 / (2008 * 2009 : ℝ)) - (2008^4 / (2009 * 2010 : ℝ))⌋ = 10 := by sorry

end floor_difference_equals_ten_l3476_347663


namespace range_of_a_l3476_347697

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (0 < a ∧ a ≤ 1/3) ∨ a > 1 :=
sorry

end range_of_a_l3476_347697


namespace milan_phone_bill_l3476_347605

/-- Calculates the number of minutes billed given the total bill, monthly fee, and per-minute rate. -/
def minutes_billed (total_bill monthly_fee per_minute_rate : ℚ) : ℚ :=
  (total_bill - monthly_fee) / per_minute_rate

/-- Proves that given the specified conditions, the number of minutes billed is 178. -/
theorem milan_phone_bill : 
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let per_minute_rate : ℚ := 0.12
  minutes_billed total_bill monthly_fee per_minute_rate = 178 := by
  sorry

end milan_phone_bill_l3476_347605


namespace inequality_and_equality_condition_l3476_347678

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3) ∧ 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ 
   a = Real.sqrt (Real.sqrt 3) ∧ b = Real.sqrt (Real.sqrt 3) ∧ c = Real.sqrt (Real.sqrt 3)) :=
by sorry

end inequality_and_equality_condition_l3476_347678


namespace no_four_digit_sum_9_div_11_l3476_347621

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem: There are no four-digit numbers whose digits add up to 9 and are divisible by 11 -/
theorem no_four_digit_sum_9_div_11 :
  ¬∃ (n : FourDigitNumber), sumOfDigits n.value = 9 ∧ n.value % 11 = 0 := by
  sorry

end no_four_digit_sum_9_div_11_l3476_347621


namespace additional_coins_for_eight_friends_l3476_347691

/-- The minimum number of additional coins needed for unique distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 1)) / 2
  if total_needed > initial_coins then
    total_needed - initial_coins
  else
    0

/-- Theorem: Given 8 friends and 28 initial coins, 8 additional coins are needed. -/
theorem additional_coins_for_eight_friends :
  min_additional_coins 8 28 = 8 := by
  sorry


end additional_coins_for_eight_friends_l3476_347691


namespace smallest_factorization_coefficient_l3476_347680

theorem smallest_factorization_coefficient (b : ℕ+) (p q : ℤ) : 
  (∀ x, (x^2 : ℤ) + b * x + 1760 = (x + p) * (x + q)) →
  (∀ b' : ℕ+, b' < b → 
    ¬∃ p' q' : ℤ, ∀ x, (x^2 : ℤ) + b' * x + 1760 = (x + p') * (x + q')) →
  b = 108 := by
sorry

end smallest_factorization_coefficient_l3476_347680


namespace system_solution_l3476_347671

theorem system_solution (a b : ℝ) : 
  (2 * 5 + b = a) ∧ (5 - 2 * b = 3) → a = 11 ∧ b = 1 := by sorry

end system_solution_l3476_347671


namespace industrial_machine_shirts_l3476_347604

/-- The number of shirts made by an industrial machine yesterday -/
def shirts_yesterday (x : ℕ) : Prop :=
  let shirts_per_minute : ℕ := 8
  let total_minutes : ℕ := 2
  let shirts_today : ℕ := 3
  let total_shirts : ℕ := shirts_per_minute * total_minutes
  x = total_shirts - shirts_today

theorem industrial_machine_shirts : shirts_yesterday 13 := by
  sorry

end industrial_machine_shirts_l3476_347604


namespace smallest_bob_number_l3476_347688

def alice_number : ℕ := 36

def has_all_prime_factors_plus_one (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ n → p ∣ m) ∧
  ∃ q : ℕ, Prime q ∧ q ∣ m ∧ ¬(q ∣ n)

theorem smallest_bob_number :
  ∃ m : ℕ, has_all_prime_factors_plus_one alice_number m ∧
  ∀ k : ℕ, has_all_prime_factors_plus_one alice_number k → m ≤ k :=
by
  sorry

end smallest_bob_number_l3476_347688


namespace pizza_slices_remaining_l3476_347612

def initial_slices : ℕ := 15
def breakfast_slices : ℕ := 4
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

theorem pizza_slices_remaining :
  initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices = 2 := by
  sorry

end pizza_slices_remaining_l3476_347612


namespace smallest_possible_c_l3476_347606

theorem smallest_possible_c (a b c : ℝ) : 
  1 < a → a < b → b < c →
  1 + a ≤ b →
  1 / a + 1 / b ≤ 1 / c →
  c ≥ (3 + Real.sqrt 5) / 2 :=
sorry

end smallest_possible_c_l3476_347606


namespace problem_solution_l3476_347649

theorem problem_solution (a b c : ℝ) 
  (h1 : (6 * a + 34) ^ (1/3 : ℝ) = 4)
  (h2 : (5 * a + b - 2) ^ (1/2 : ℝ) = 5)
  (h3 : c = 9 ^ (1/2 : ℝ)) :
  a = 5 ∧ b = 2 ∧ c = 3 ∧ (3 * a - b + c) ^ (1/2 : ℝ) = 4 ∨ (3 * a - b + c) ^ (1/2 : ℝ) = -4 :=
by sorry

end problem_solution_l3476_347649


namespace max_value_2x_3y_l3476_347646

theorem max_value_2x_3y (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) :
  ∃ (M : ℝ), M = Real.sqrt 31 ∧ 2*x + 3*y ≤ M ∧ ∀ (N : ℝ), (∀ (a b : ℝ), 3 * a^2 + b^2 ≤ 3 → 2*a + 3*b ≤ N) → M ≤ N :=
sorry

end max_value_2x_3y_l3476_347646


namespace complement_M_intersect_N_l3476_347618

-- Define the sets M and N
def M : Set ℝ := {x | -4 < x - 1 ∧ x - 1 ≤ 4}
def N : Set ℝ := {x | x^2 < 25}

-- Define the complement of M in ℝ
def C_R_M : Set ℝ := {x | x ∉ M}

-- State the theorem
theorem complement_M_intersect_N :
  (C_R_M ∩ N) = {x : ℝ | -5 < x ∧ x ≤ -3} := by sorry

end complement_M_intersect_N_l3476_347618


namespace inequality_proof_l3476_347643

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
sorry

end inequality_proof_l3476_347643


namespace triangle_min_angle_le_60_l3476_347630

theorem triangle_min_angle_le_60 (α β γ : ℝ) :
  α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0 → min α (min β γ) ≤ 60 := by
  sorry

end triangle_min_angle_le_60_l3476_347630


namespace quadratic_inequality_l3476_347622

theorem quadratic_inequality (x : ℝ) : x^2 - 50*x + 625 ≤ 25 ↔ 20 ≤ x ∧ x ≤ 30 := by
  sorry

end quadratic_inequality_l3476_347622


namespace total_full_spots_l3476_347673

/-- Calculates the number of full parking spots in a multi-story parking garage -/
def fullParkingSpots : ℕ :=
  let totalLevels : ℕ := 7
  let firstLevelSpots : ℕ := 100
  let spotIncrease : ℕ := 50
  let firstLevelOpenSpots : ℕ := 58
  let openSpotDecrease : ℕ := 3
  let openSpotIncrease : ℕ := 10
  let switchLevel : ℕ := 4

  let totalFullSpots : ℕ := (List.range totalLevels).foldl
    (fun acc level =>
      let totalSpots := firstLevelSpots + level * spotIncrease
      let openSpots := if level < switchLevel - 1
        then firstLevelOpenSpots - level * openSpotDecrease
        else firstLevelOpenSpots - (switchLevel - 1) * openSpotDecrease + (level - switchLevel + 1) * openSpotIncrease
      acc + (totalSpots - openSpots))
    0

  totalFullSpots

/-- The theorem stating that the total number of full parking spots is 1329 -/
theorem total_full_spots : fullParkingSpots = 1329 := by
  sorry

end total_full_spots_l3476_347673


namespace min_value_reciprocal_sum_l3476_347625

/-- Given positive real numbers m and n, and perpendicular vectors (m, 1) and (1, n-1),
    the minimum value of 1/m + 2/n is 3 + 2√2. -/
theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
    (h_perp : m * 1 + 1 * (n - 1) = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * 1 + 1 * (y - 1) = 0 → 1/x + 2/y ≥ 1/m + 2/n) →
  1/m + 2/n = 3 + 2 * Real.sqrt 2 := by
sorry

end min_value_reciprocal_sum_l3476_347625


namespace sum_of_ages_l3476_347653

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71 years. -/
theorem sum_of_ages (beckett olaf shannen jack : ℕ) : 
  beckett = 12 ∧ 
  olaf = beckett + 3 ∧ 
  shannen = olaf - 2 ∧ 
  jack = 2 * shannen + 5 → 
  beckett + olaf + shannen + jack = 71 := by
sorry

end sum_of_ages_l3476_347653


namespace sum_of_digits_square_22222_l3476_347626

/-- The sum of the digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The square of 22222 -/
def square_22222 : ℕ := 22222 * 22222

theorem sum_of_digits_square_22222 : sum_of_digits square_22222 = 46 := by
  sorry

end sum_of_digits_square_22222_l3476_347626


namespace statement_d_not_always_true_l3476_347642

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line)
variable (α β : Plane)
variable (h1 : ¬ parallel m n)
variable (h2 : α ≠ β)

-- State the theorem
theorem statement_d_not_always_true :
  ¬ (∀ (m : Line) (α β : Plane),
    plane_perpendicular α β →
    contained_in m α →
    perpendicular m β) :=
by sorry

end statement_d_not_always_true_l3476_347642
