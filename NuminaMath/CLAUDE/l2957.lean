import Mathlib

namespace jessica_rearrangement_time_l2957_295747

/-- The time in hours required to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (repeated_letter_count : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  let total_permutations := (name_length.factorial / repeated_letter_count.factorial : ℚ)
  let time_in_minutes := total_permutations / rearrangements_per_minute
  time_in_minutes / 60

/-- Theorem stating the time required to write all rearrangements of Jessica's name -/
theorem jessica_rearrangement_time :
  time_to_write_rearrangements 7 2 18 = 2333 / 1000 := by
  sorry

end jessica_rearrangement_time_l2957_295747


namespace cost_price_determination_l2957_295792

theorem cost_price_determination (loss_percentage : Real) (gain_percentage : Real) (price_increase : Real) :
  loss_percentage = 0.1 →
  gain_percentage = 0.1 →
  price_increase = 50 →
  ∃ (cost_price : Real),
    cost_price * (1 - loss_percentage) + price_increase = cost_price * (1 + gain_percentage) ∧
    cost_price = 250 :=
by sorry

end cost_price_determination_l2957_295792


namespace certain_term_is_12th_l2957_295796

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  -- Sum of a certain term and the 12th term is 20
  certain_term_sum : ∃ n : ℕ, a + (n - 1) * d + (a + 11 * d) = 20
  -- Sum of first 12 terms is 120
  sum_12_terms : 6 * (2 * a + 11 * d) = 120

/-- The certain term is the 12th term itself -/
theorem certain_term_is_12th (ap : ArithmeticProgression) : 
  ∃ n : ℕ, n = 12 ∧ a + (n - 1) * d + (a + 11 * d) = 20 := by
  sorry

#check certain_term_is_12th

end certain_term_is_12th_l2957_295796


namespace fraction_equality_l2957_295713

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 37) = 875 / 1000 → a = 259 := by
  sorry

end fraction_equality_l2957_295713


namespace function_not_in_first_quadrant_l2957_295733

theorem function_not_in_first_quadrant (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x : ℝ, x > 0 → a^x + b < 0 := by sorry

end function_not_in_first_quadrant_l2957_295733


namespace math_statements_l2957_295700

theorem math_statements :
  (8^0 = 1) ∧
  (|-8| = 8) ∧
  (-(-8) = 8) ∧
  (¬(Real.sqrt 8 = 2 * Real.sqrt 2 ∨ Real.sqrt 8 = -2 * Real.sqrt 2)) := by
  sorry

end math_statements_l2957_295700


namespace quadratic_equations_common_root_l2957_295721

theorem quadratic_equations_common_root (p q r s : ℝ) 
  (hq : q ≠ -1) (hs : s ≠ -1) : 
  (∃ (a b : ℝ), (a^2 + p*a + q = 0 ∧ a^2 + r*a + s = 0) ∧ 
   (b^2 + p*b + q = 0 ∧ (1/b)^2 + r*(1/b) + s = 0)) ↔ 
  (p*r = (q+1)*(s+1) ∧ p*(q+1)*s = r*(s+1)*q) :=
sorry

end quadratic_equations_common_root_l2957_295721


namespace f_derivative_l2957_295758

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem f_derivative (x : ℝ) (hx : x ≠ 0) :
  deriv f x = (Real.exp x * (x - 1)) / (x^2) :=
by sorry

end f_derivative_l2957_295758


namespace smallest_solution_of_equation_l2957_295767

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 3 * x / (x - 3) + (3 * x^2 - 27) / x
  ∃ (min_sol : ℝ), min_sol = (8 - Real.sqrt 145) / 3 ∧
    f min_sol = 14 ∧
    ∀ (y : ℝ), f y = 14 → y ≥ min_sol :=
by sorry

end smallest_solution_of_equation_l2957_295767


namespace radical_expression_simplification_l2957_295780

theorem radical_expression_simplification
  (a b x : ℝ) 
  (h1 : a < b) 
  (h2 : -b ≤ x) 
  (h3 : x ≤ -a) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * Real.sqrt (-(x + a) * (x + b)) :=
by sorry

end radical_expression_simplification_l2957_295780


namespace sum_of_roots_satisfies_equation_l2957_295725

-- Define the polynomial
def polynomial (a b c x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

-- Define the equation for the sum of two roots
def sum_of_roots_equation (a b c u : ℝ) : ℝ := u^6 + 2*a*u^4 + (a^2 - 4*c)*u^2 - b^2

-- Theorem statement
theorem sum_of_roots_satisfies_equation (a b c : ℝ) :
  ∃ (x₁ x₂ : ℝ), polynomial a b c x₁ = 0 ∧ polynomial a b c x₂ = 0 ∧
  (∃ (u : ℝ), u = x₁ + x₂ ∧ sum_of_roots_equation a b c u = 0) :=
by sorry

end sum_of_roots_satisfies_equation_l2957_295725


namespace moon_weight_calculation_l2957_295755

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The weight of Mars in tons -/
def mars_weight : ℝ := 500

/-- The percentage of iron in the composition -/
def iron_percentage : ℝ := 50

/-- The percentage of carbon in the composition -/
def carbon_percentage : ℝ := 20

/-- The percentage of other elements in the composition -/
def other_percentage : ℝ := 100 - iron_percentage - carbon_percentage

/-- The weight of other elements on Mars in tons -/
def mars_other_elements : ℝ := 150

theorem moon_weight_calculation :
  moon_weight = mars_weight / 2 ∧
  mars_weight = mars_other_elements / (other_percentage / 100) :=
by sorry

end moon_weight_calculation_l2957_295755


namespace lunch_spending_difference_l2957_295783

/-- Given a lunch scenario where two people spent a total of $15,
    with one person spending $10, prove that the difference in
    spending between the two people is $5. -/
theorem lunch_spending_difference :
  ∀ (your_spending friend_spending : ℕ),
  your_spending + friend_spending = 15 →
  friend_spending = 10 →
  friend_spending > your_spending →
  friend_spending - your_spending = 5 :=
by
  sorry

end lunch_spending_difference_l2957_295783


namespace derivative_of_power_function_l2957_295761

theorem derivative_of_power_function (a k : ℝ) (x : ℝ) :
  deriv (λ x => (3 * a * x - x^2)^k) x = k * (3 * a - 2 * x) * (3 * a * x - x^2)^(k - 1) :=
sorry

end derivative_of_power_function_l2957_295761


namespace odd_function_symmetry_l2957_295746

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y
def hasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop := 
  ∀ x, a ≤ x ∧ x ≤ b → f x ≥ m

-- State the theorem
theorem odd_function_symmetry (hOdd : isOdd f) 
  (hDec : isDecreasingOn f (-2) (-1)) 
  (hMin : hasMinimumOn f (-2) (-1) 3) :
  isDecreasingOn f 1 2 ∧ hasMinimumOn f 1 2 (-3) := by
  sorry

end odd_function_symmetry_l2957_295746


namespace graphics_cards_sold_l2957_295741

/-- Represents the number of graphics cards sold. -/
def graphics_cards : ℕ := sorry

/-- Represents the number of hard drives sold. -/
def hard_drives : ℕ := 14

/-- Represents the number of CPUs sold. -/
def cpus : ℕ := 8

/-- Represents the number of RAM pairs sold. -/
def ram_pairs : ℕ := 4

/-- Represents the price of a single graphics card in dollars. -/
def graphics_card_price : ℕ := 600

/-- Represents the price of a single hard drive in dollars. -/
def hard_drive_price : ℕ := 80

/-- Represents the price of a single CPU in dollars. -/
def cpu_price : ℕ := 200

/-- Represents the price of a pair of RAM in dollars. -/
def ram_pair_price : ℕ := 60

/-- Represents the total earnings of the store in dollars. -/
def total_earnings : ℕ := 8960

/-- Theorem stating that the number of graphics cards sold is 10. -/
theorem graphics_cards_sold : graphics_cards = 10 := by
  sorry

end graphics_cards_sold_l2957_295741


namespace geometric_progression_ratio_equation_l2957_295723

/-- Given distinct non-zero real numbers x, y, and z, and a real number r,
    if x^2(y-z), y^2(z-x), and z^2(x-y) form a geometric progression with common ratio r,
    then r satisfies the equation r^2 + r + 1 = 0 -/
theorem geometric_progression_ratio_equation (x y z r : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (hprogression : ∃ (a : ℝ), a ≠ 0 ∧ 
    x^2 * (y - z) = a ∧ 
    y^2 * (z - x) = a * r ∧ 
    z^2 * (x - y) = a * r^2) :
  r^2 + r + 1 = 0 := by sorry

end geometric_progression_ratio_equation_l2957_295723


namespace hawk_crow_percentage_l2957_295719

theorem hawk_crow_percentage (num_crows : ℕ) (total_birds : ℕ) (percentage : ℚ) : 
  num_crows = 30 →
  total_birds = 78 →
  total_birds = num_crows + (num_crows * (1 + percentage / 100)) →
  percentage = 60 := by
sorry

end hawk_crow_percentage_l2957_295719


namespace min_values_ab_and_a_plus_2b_l2957_295727

theorem min_values_ab_and_a_plus_2b (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 2) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 2/b = 2 → a * b ≥ 2) ∧ 
  (∀ a b, a > 0 → b > 0 → 1/a + 2/b = 2 → a + 2*b ≥ 9/2) ∧
  (a = 3/2 ∧ b = 3/2 → a * b = 2 ∧ a + 2*b = 9/2) :=
sorry

end min_values_ab_and_a_plus_2b_l2957_295727


namespace one_point_six_million_scientific_notation_l2957_295717

theorem one_point_six_million_scientific_notation :
  (1.6 : ℝ) * (1000000 : ℝ) = (1.6 : ℝ) * (10 : ℝ) ^ 6 := by
  sorry

end one_point_six_million_scientific_notation_l2957_295717


namespace alien_arms_count_l2957_295726

/-- The number of arms an alien has -/
def alien_arms : ℕ := sorry

/-- The number of legs an alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

theorem alien_arms_count : alien_arms = 3 :=
  by
    have h1 : 5 * (alien_arms + alien_legs) = 5 * (martian_arms + martian_legs) + 5 := by sorry
    sorry

end alien_arms_count_l2957_295726


namespace complex_number_equality_l2957_295774

theorem complex_number_equality (z : ℂ) : z = 2 - (13 / 6) * I →
  Complex.abs (z - 2) = Complex.abs (z + 2) ∧
  Complex.abs (z - 2) = Complex.abs (z - 3 * I) := by
  sorry

end complex_number_equality_l2957_295774


namespace f_at_two_l2957_295751

def f (x : ℝ) : ℝ := 15 * x^5 - 24 * x^4 + 33 * x^3 - 42 * x^2 + 51 * x

theorem f_at_two : f 2 = 294 := by
  sorry

end f_at_two_l2957_295751


namespace max_ratio_squared_l2957_295757

theorem max_ratio_squared (a b x z : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx : 0 ≤ x) (hxa : x < a) (hz : 0 ≤ z) (hzb : z < b)
  (heq : a^2 + z^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - z)^2) :
  (a / b)^2 ≤ 4/3 :=
by sorry

end max_ratio_squared_l2957_295757


namespace circle_center_l2957_295787

/-- The center of a circle given by the equation x^2 - 4x + y^2 - 6y - 12 = 0 is (2, 3) -/
theorem circle_center (x y : ℝ) : 
  x^2 - 4*x + y^2 - 6*y - 12 = 0 → (2, 3) = (x, y) := by
  sorry

end circle_center_l2957_295787


namespace sum_less_than_addends_implies_negative_l2957_295772

theorem sum_less_than_addends_implies_negative (a b : ℝ) :
  (a + b < a ∧ a + b < b) → (a < 0 ∧ b < 0) := by
  sorry

end sum_less_than_addends_implies_negative_l2957_295772


namespace parabola_directrix_l2957_295739

open Real

-- Define the parabola
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

-- Define points
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problem_setup (C : Parabola) (O F P Q : Point) : Prop :=
  -- O is the coordinate origin
  O.x = 0 ∧ O.y = 0
  -- F is the focus of parabola C
  ∧ F.x = C.p/2 ∧ F.y = 0
  -- P is a point on C
  ∧ C.eq P.x P.y
  -- PF is perpendicular to the x-axis
  ∧ P.x = F.x
  -- Q is a point on the x-axis
  ∧ Q.y = 0
  -- PQ is perpendicular to OP
  ∧ (Q.y - P.y) * (P.x - O.x) + (Q.x - P.x) * (P.y - O.y) = 0
  -- |FQ| = 6
  ∧ |F.x - Q.x| = 6

-- Theorem statement
theorem parabola_directrix (C : Parabola) (O F P Q : Point) 
  (h : problem_setup C O F P Q) : 
  ∃ (x : ℝ), x = -3/2 ∧ ∀ (y : ℝ), C.eq x y ↔ False :=
sorry

end parabola_directrix_l2957_295739


namespace min_value_trigonometric_fraction_l2957_295709

theorem min_value_trigonometric_fraction (a b : ℝ) (θ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hθ : θ ∈ Set.Ioo 0 (π / 2)) :
  a / Real.sin θ + b / Real.cos θ ≥ (Real.rpow a (2/3) + Real.rpow b (2/3))^2 := by
  sorry

end min_value_trigonometric_fraction_l2957_295709


namespace download_time_proof_l2957_295714

/-- Proves that the download time for a 360 GB program at 50 MB/s is 2 hours -/
theorem download_time_proof (download_speed : ℝ) (program_size : ℝ) (mb_per_gb : ℝ) :
  download_speed = 50 ∧ program_size = 360 ∧ mb_per_gb = 1000 →
  (program_size * mb_per_gb) / (download_speed * 3600) = 2 := by
  sorry

end download_time_proof_l2957_295714


namespace symmetric_point_coordinates_l2957_295742

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetricXAxis (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- The given point N. -/
def N : Point2D :=
  ⟨2, 3⟩

theorem symmetric_point_coordinates :
  symmetricXAxis N = ⟨2, -3⟩ := by
  sorry

end symmetric_point_coordinates_l2957_295742


namespace gcf_of_7_factorial_and_8_factorial_l2957_295745

theorem gcf_of_7_factorial_and_8_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcf_of_7_factorial_and_8_factorial_l2957_295745


namespace jack_afternoon_emails_l2957_295735

/-- The number of emails Jack received in a day -/
def total_emails : ℕ := 10

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 1

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := total_emails - morning_emails - evening_emails

theorem jack_afternoon_emails :
  afternoon_emails = 4 := by sorry

end jack_afternoon_emails_l2957_295735


namespace gift_bags_production_time_l2957_295768

theorem gift_bags_production_time (total_bags : ℕ) (rate_per_day : ℕ) (h1 : total_bags = 519) (h2 : rate_per_day = 42) :
  (total_bags + rate_per_day - 1) / rate_per_day = 13 :=
sorry

end gift_bags_production_time_l2957_295768


namespace application_methods_count_l2957_295737

def number_of_universities : ℕ := 6
def universities_to_choose : ℕ := 3
def universities_with_conflict : ℕ := 2

theorem application_methods_count :
  (number_of_universities.choose universities_to_choose) -
  (universities_with_conflict * (number_of_universities - universities_with_conflict).choose (universities_to_choose - 1)) = 16 := by
  sorry

end application_methods_count_l2957_295737


namespace rational_inequality_solution_l2957_295716

theorem rational_inequality_solution (x : ℝ) : 
  (x + 2) / (x^2 + 3*x + 10) ≥ 0 ↔ x ≥ -2 := by sorry

end rational_inequality_solution_l2957_295716


namespace trigonometric_identity_equivalence_l2957_295743

theorem trigonometric_identity_equivalence (x : ℝ) :
  (1 + Real.cos (4 * x)) * Real.sin (2 * x) = (Real.cos (2 * x))^2 ↔
  (∃ k : ℤ, x = (-1)^k * (π / 12) + k * (π / 2)) ∨
  (∃ n : ℤ, x = π / 4 * (2 * n + 1)) := by
  sorry

end trigonometric_identity_equivalence_l2957_295743


namespace max_value_problem_l2957_295730

theorem max_value_problem (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ Real.sqrt (2292.25 / 225) :=
sorry

end max_value_problem_l2957_295730


namespace equation_is_linear_and_has_solution_l2957_295764

-- Define the equation
def equation (x : ℝ) : Prop := 1 - x = -3

-- State the theorem
theorem equation_is_linear_and_has_solution :
  (∃ a b : ℝ, ∀ x, equation x ↔ a * x + b = 0) ∧ 
  equation 4 := by sorry

end equation_is_linear_and_has_solution_l2957_295764


namespace price_per_large_bottle_l2957_295785

/-- The price per large bottle, given the number of large and small bottles,
    the price of small bottles, and the average price of all bottles. -/
theorem price_per_large_bottle (large_count small_count : ℕ)
                                (small_price avg_price : ℚ) :
  large_count = 1325 →
  small_count = 750 →
  small_price = 138/100 →
  avg_price = 17057/10000 →
  ∃ (large_price : ℚ), 
    (large_count * large_price + small_count * small_price) / (large_count + small_count) = avg_price ∧
    abs (large_price - 189/100) < 1/100 := by
  sorry

end price_per_large_bottle_l2957_295785


namespace sum_of_three_numbers_l2957_295793

theorem sum_of_three_numbers (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 60) (h_xz : x * z = 90) (h_yz : y * z = 150) : 
  x + y + z = 31 := by
sorry

end sum_of_three_numbers_l2957_295793


namespace equation_solutions_l2957_295731

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x - 1)^2 - 4*x = 0
def equation2 (x : ℝ) : Prop := (2*x - 3)^2 = x^2

-- State the theorem
theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 3 / 2 ∧ x2 = 1 - Real.sqrt 3 / 2 ∧ equation1 x1 ∧ equation1 x2) ∧
  (∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 1 ∧ equation2 x1 ∧ equation2 x2) :=
by sorry

end equation_solutions_l2957_295731


namespace proposition_1_proposition_2_proposition_3_no_false_main_theorem_l2957_295795

-- Proposition 1
theorem proposition_1 (k : ℝ) : k > 0 → ∃ x : ℝ, x^2 - 2*x - k = 0 := by sorry

-- Proposition 2
theorem proposition_2 (x y : ℝ) : x + y ≠ 8 → x ≠ 2 ∨ y ≠ 6 := by sorry

-- Proposition 3
theorem proposition_3_no_false : ¬∃ (P : Prop), P ↔ ¬(∀ x y : ℝ, x*y = 0 → x = 0 ∨ y = 0) := by sorry

-- Main theorem combining all propositions
theorem main_theorem : 
  (∀ k : ℝ, k > 0 → ∃ x : ℝ, x^2 - 2*x - k = 0) ∧ 
  (∀ x y : ℝ, x + y ≠ 8 → x ≠ 2 ∨ y ≠ 6) ∧ 
  ¬∃ (P : Prop), P ↔ ¬(∀ x y : ℝ, x*y = 0 → x = 0 ∨ y = 0) := by sorry

end proposition_1_proposition_2_proposition_3_no_false_main_theorem_l2957_295795


namespace current_library_books_l2957_295708

def library_books (initial : ℕ) (first_purchase : ℕ) (second_purchase : ℕ) (donation : ℕ) : ℕ :=
  initial + first_purchase + second_purchase - donation

theorem current_library_books :
  library_books 500 300 400 200 = 1000 := by sorry

end current_library_books_l2957_295708


namespace intersection_of_A_and_B_l2957_295722

-- Define the sets A and B
def A : Set ℝ := {x | x / (x - 1) < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2957_295722


namespace not_prime_n4_2n2_3_l2957_295778

theorem not_prime_n4_2n2_3 (n : ℤ) : ∃ k : ℤ, n^4 + 2*n^2 + 3 = 3 * k := by
  sorry

end not_prime_n4_2n2_3_l2957_295778


namespace unique_triplet_satisfying_conditions_l2957_295775

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c : ℝ),
    ({a^2 - 4*c, b^2 - 2*a, c^2 - 2*b} : Set ℝ) = {a - c, b - 4*c, a + b} ∧
    2*a + 2*b + 6 = 5*c ∧
    (a^2 - 4*c ≠ b^2 - 2*a ∧ a^2 - 4*c ≠ c^2 - 2*b ∧ b^2 - 2*a ≠ c^2 - 2*b) ∧
    (a - c ≠ b - 4*c ∧ a - c ≠ a + b ∧ b - 4*c ≠ a + b) ∧
    a = 1 ∧ b = 1 ∧ c = 2 := by
  sorry

end unique_triplet_satisfying_conditions_l2957_295775


namespace characterization_of_solutions_l2957_295711

/-- Sum of digits of a non-negative integer -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The set of solutions -/
def solution_set : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19}

/-- Main theorem: n ≤ 2s(n) iff n is in the solution set -/
theorem characterization_of_solutions (n : ℕ) :
  n ≤ 2 * sum_of_digits n ↔ n ∈ solution_set :=
sorry

end characterization_of_solutions_l2957_295711


namespace florist_roses_theorem_l2957_295777

/-- Calculates the final number of roses a florist has after selling and picking more roses. -/
def final_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Proves that the final number of roses is correct given the initial number,
    the number sold, and the number picked. -/
theorem florist_roses_theorem (initial : ℕ) (sold : ℕ) (picked : ℕ) 
    (h1 : initial ≥ sold) : 
  final_roses initial sold picked = initial - sold + picked :=
by
  -- The proof goes here
  sorry

/-- Verifies the specific case from the original problem. -/
example : final_roses 37 16 19 = 40 :=
by
  -- The proof goes here
  sorry

end florist_roses_theorem_l2957_295777


namespace log_power_difference_l2957_295712

theorem log_power_difference (x : ℝ) (h1 : x < 1) 
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^2) / Real.log 10 = 48) :
  (Real.log x / Real.log 10)^5 - Real.log (x^5) / Real.log 10 = -7746 := by
  sorry

end log_power_difference_l2957_295712


namespace remove_seven_maintain_coverage_l2957_295781

/-- Represents a collection of objects covering a surface -/
structure CoveringSet (n : ℕ) :=
  (area : ℝ)
  (total_coverage : ℝ)
  (coverage : Fin n → ℝ)
  (covers_completely : total_coverage = area)
  (non_negative_coverage : ∀ i, coverage i ≥ 0)
  (sum_coverage : (Finset.sum Finset.univ coverage) = total_coverage)

/-- Theorem stating that it's possible to remove 7 objects from a set of 15
    such that the remaining 8 cover at least 8/15 of the total area -/
theorem remove_seven_maintain_coverage 
  (s : CoveringSet 15) : 
  ∃ (removed : Finset (Fin 15)), 
    Finset.card removed = 7 ∧ 
    (Finset.sum (Finset.univ \ removed) s.coverage) ≥ (8/15) * s.area := by
  sorry

end remove_seven_maintain_coverage_l2957_295781


namespace janice_starting_sentences_janice_started_with_258_sentences_l2957_295732

/-- Calculates the number of sentences Janice started with today -/
theorem janice_starting_sentences 
  (typing_speed : ℕ) 
  (typing_duration1 typing_duration2 typing_duration3 : ℕ)
  (erased_sentences : ℕ)
  (total_sentences : ℕ) : ℕ :=
  let total_duration := typing_duration1 + typing_duration2 + typing_duration3
  let typed_sentences := total_duration * typing_speed
  let net_typed_sentences := typed_sentences - erased_sentences
  total_sentences - net_typed_sentences

/-- Proves that Janice started with 258 sentences today -/
theorem janice_started_with_258_sentences : 
  janice_starting_sentences 6 20 15 18 40 536 = 258 := by
  sorry

end janice_starting_sentences_janice_started_with_258_sentences_l2957_295732


namespace davids_math_marks_l2957_295710

/-- Represents the marks obtained in each subject -/
structure SubjectMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks given the total marks and number of subjects -/
def average (total : ℕ) (subjects : ℕ) : ℚ :=
  (total : ℚ) / (subjects : ℚ)

/-- Theorem stating that David's marks in Mathematics are 35 -/
theorem davids_math_marks (marks : SubjectMarks) 
    (h1 : marks.english = 36)
    (h2 : marks.physics = 42)
    (h3 : marks.chemistry = 57)
    (h4 : marks.biology = 55)
    (h5 : average (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) 5 = 45) :
    marks.mathematics = 35 := by
  sorry

#check davids_math_marks

end davids_math_marks_l2957_295710


namespace cubic_polynomial_problem_l2957_295704

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := x^3 + 4*x^2 + 7*x + 10

-- Define the roots a, b, c
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Define P(x)
noncomputable def P : ℝ → ℝ := sorry

-- Theorem statement
theorem cubic_polynomial_problem :
  (cubic_equation a = 0) ∧ 
  (cubic_equation b = 0) ∧ 
  (cubic_equation c = 0) ∧
  (P a = 2*(b + c)) ∧
  (P b = 2*(a + c)) ∧
  (P c = 2*(a + b)) ∧
  (P (a + b + c) = -20) →
  ∀ x, P x = (4*x^3 + 16*x^2 + 55*x - 16) / 9 := by
  sorry

end cubic_polynomial_problem_l2957_295704


namespace unique_solution_implies_equal_absolute_values_l2957_295749

theorem unique_solution_implies_equal_absolute_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x, a * (x - a)^2 + b * (x - b)^2 = 0) → |a| = |b| :=
by sorry

end unique_solution_implies_equal_absolute_values_l2957_295749


namespace equation_solution_l2957_295724

theorem equation_solution :
  ∃ x : ℚ, x ≠ 2 ∧ (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ∧ x = -2/3 := by
  sorry

end equation_solution_l2957_295724


namespace law_of_sines_l2957_295718

/-- The Law of Sines for a triangle ABC -/
theorem law_of_sines (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) :=
sorry

end law_of_sines_l2957_295718


namespace cafeteria_bags_l2957_295786

theorem cafeteria_bags (total : ℕ) (x : ℕ) : 
  total = 351 → 
  (x + 20) - 3 * ((total - x) - 50) = 1 → 
  x = 221 ∧ (total - x) = 130 := by
  sorry

end cafeteria_bags_l2957_295786


namespace pole_length_reduction_l2957_295759

theorem pole_length_reduction (original_length current_length : ℝ) 
  (h1 : original_length = 20)
  (h2 : current_length = 14) :
  (original_length - current_length) / original_length * 100 = 30 := by
sorry

end pole_length_reduction_l2957_295759


namespace lcm_problem_l2957_295760

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 12 := by
  sorry

end lcm_problem_l2957_295760


namespace complement_of_A_in_U_l2957_295762

def U : Set ℕ := {x | x ≥ 2}
def A : Set ℕ := {x | x^2 ≥ 5}

theorem complement_of_A_in_U : (U \ A) = {2} := by sorry

end complement_of_A_in_U_l2957_295762


namespace rectangle_width_is_six_l2957_295744

/-- A rectangle with given properties -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ
  diagonal_squares : ℕ

/-- The properties of our specific rectangle -/
def my_rectangle : Rectangle where
  length := 8
  width := 6
  area := 48
  diagonal_squares := 12

/-- Theorem stating that the width of the rectangle is 6 inches -/
theorem rectangle_width_is_six (r : Rectangle) 
  (h1 : r.length = 8)
  (h2 : r.area = 48)
  (h3 : r.diagonal_squares = 12) : 
  r.width = 6 := by
  sorry

#check rectangle_width_is_six

end rectangle_width_is_six_l2957_295744


namespace no_nonneg_integer_solution_l2957_295794

theorem no_nonneg_integer_solution :
  ¬ ∃ y : ℕ, Real.sqrt ((y - 2)^2 + 4^2) = 7 := by
  sorry

end no_nonneg_integer_solution_l2957_295794


namespace remainder_theorem_l2957_295784

theorem remainder_theorem : (2^210 + 210) % (2^105 + 2^63 + 1) = 210 := by
  sorry

end remainder_theorem_l2957_295784


namespace complex_modulus_problem_l2957_295707

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 1 + 2*I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
sorry

end complex_modulus_problem_l2957_295707


namespace car_speed_acceleration_l2957_295736

/-- Proves that given an initial speed of 45 m/s, an acceleration of 2.5 m/s² for 10 seconds,
    the final speed will be 70 m/s and 252 km/h. -/
theorem car_speed_acceleration (initial_speed : Real) (acceleration : Real) (time : Real) :
  initial_speed = 45 ∧ acceleration = 2.5 ∧ time = 10 →
  let final_speed := initial_speed + acceleration * time
  final_speed = 70 ∧ final_speed * 3.6 = 252 := by
  sorry

end car_speed_acceleration_l2957_295736


namespace percentage_relation_l2957_295765

theorem percentage_relation (x y : ℝ) (h : 0.6 * (x - y) = 0.2 * (x + y)) : y = 0.5 * x := by
  sorry

end percentage_relation_l2957_295765


namespace monomial_properties_l2957_295752

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial (R : Type*) [CommRing R] where
  coeff : R
  exponents : List ℕ

/-- The degree of a monomial is the sum of its exponents. -/
def Monomial.degree {R : Type*} [CommRing R] (m : Monomial R) : ℕ :=
  m.exponents.sum

/-- Our specific monomial -3x^2y -/
def our_monomial : Monomial ℤ :=
  { coeff := -3
  , exponents := [2, 1] }

theorem monomial_properties :
  our_monomial.coeff = -3 ∧ our_monomial.degree = 3 := by
  sorry

end monomial_properties_l2957_295752


namespace six_digit_divisibility_l2957_295771

theorem six_digit_divisibility (A B : ℕ) 
  (hA : A ≥ 100 ∧ A < 1000) 
  (hB : B ≥ 100 ∧ B < 1000) 
  (hAnotDiv : ¬ (37 ∣ A)) 
  (hBnotDiv : ¬ (37 ∣ B)) 
  (hSum : 37 ∣ (A + B)) : 
  37 ∣ (1000 * A + B) := by
  sorry

end six_digit_divisibility_l2957_295771


namespace shop_prices_existence_l2957_295705

theorem shop_prices_existence (S : ℕ) (h : S ≥ 100) :
  ∃ (a b c P : ℕ), 
    a > b ∧ b > c ∧ 
    a + b + c = S ∧
    a * b * c = P ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∃ (a' b' c' : ℕ), (a' ≠ a ∨ b' ≠ b ∨ c' ≠ c) ∧
      a' > b' ∧ b' > c' ∧
      a' + b' + c' = S ∧
      a' * b' * c' = P ∧
      a' > 0 ∧ b' > 0 ∧ c' > 0 :=
by sorry

end shop_prices_existence_l2957_295705


namespace plane_equation_proof_l2957_295715

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of a plane in 3D space -/
structure PlaneEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given points A, B, and C, proves that the equation x + 2y + 4z - 5 = 0
    represents the plane passing through point A and perpendicular to vector BC -/
theorem plane_equation_proof 
  (A : Point3D) 
  (B : Point3D) 
  (C : Point3D) 
  (h1 : A.x = -7 ∧ A.y = 0 ∧ A.z = 3)
  (h2 : B.x = 1 ∧ B.y = -5 ∧ B.z = -4)
  (h3 : C.x = 2 ∧ C.y = -3 ∧ C.z = 0) :
  let BC : Vector3D := ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩
  let plane : PlaneEquation := ⟨1, 2, 4, -5⟩
  (plane.a * (A.x - x) + plane.b * (A.y - y) + plane.c * (A.z - z) = 0) ∧
  (plane.a * BC.x + plane.b * BC.y + plane.c * BC.z = 0) :=
by sorry


end plane_equation_proof_l2957_295715


namespace advertising_agency_clients_l2957_295799

theorem advertising_agency_clients (total : ℕ) (tv radio mag tv_mag tv_radio radio_mag : ℕ) 
  (h_total : total = 180)
  (h_tv : tv = 115)
  (h_radio : radio = 110)
  (h_mag : mag = 130)
  (h_tv_mag : tv_mag = 85)
  (h_tv_radio : tv_radio = 75)
  (h_radio_mag : radio_mag = 95) :
  total = tv + radio + mag - tv_mag - tv_radio - radio_mag + 80 :=
sorry

end advertising_agency_clients_l2957_295799


namespace gcd_9009_14014_l2957_295790

theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  sorry

end gcd_9009_14014_l2957_295790


namespace problem_1_l2957_295766

theorem problem_1 (m n : ℤ) (h1 : 4*m + n = 90) (h2 : 2*m - 3*n = 10) :
  (m + 2*n)^2 - (3*m - n)^2 = -900 := by
  sorry

end problem_1_l2957_295766


namespace aquafaba_needed_l2957_295750

/-- The number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- The number of cakes Christine is making -/
def num_cakes : ℕ := 2

/-- The number of egg whites required for each cake -/
def egg_whites_per_cake : ℕ := 8

/-- Theorem stating the total number of tablespoons of aquafaba needed -/
theorem aquafaba_needed : 
  aquafaba_per_egg * num_cakes * egg_whites_per_cake = 32 := by
  sorry

end aquafaba_needed_l2957_295750


namespace exists_distinct_diagonal_products_l2957_295728

/-- A type representing the vertices of a nonagon -/
inductive Vertex : Type
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9

/-- A function type representing an arrangement of numbers on the nonagon vertices -/
def Arrangement := Vertex → Fin 9

/-- The set of all diagonals in a nonagon -/
def Diagonals : Set (Vertex × Vertex) := sorry

/-- Calculate the product of numbers at the ends of a diagonal -/
def diagonalProduct (arr : Arrangement) (d : Vertex × Vertex) : Nat := sorry

/-- Theorem stating that there exists an arrangement with all distinct diagonal products -/
theorem exists_distinct_diagonal_products :
  ∃ (arr : Arrangement), Function.Injective (diagonalProduct arr) := by sorry

end exists_distinct_diagonal_products_l2957_295728


namespace part_to_whole_ratio_l2957_295701

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 240) (h2 : (1/5) * N + 6 = P - 6) :
  (P - 6) / N = 9 / 40 := by
  sorry

end part_to_whole_ratio_l2957_295701


namespace smallest_room_length_l2957_295769

/-- Given two rectangular rooms, where the larger room has dimensions 45 feet by 30 feet,
    and the smaller room has a width of 15 feet, if the difference in area between
    these two rooms is 1230 square feet, then the length of the smaller room is 8 feet. -/
theorem smallest_room_length
  (larger_width : ℝ) (larger_length : ℝ)
  (smaller_width : ℝ) (smaller_length : ℝ)
  (area_difference : ℝ) :
  larger_width = 45 →
  larger_length = 30 →
  smaller_width = 15 →
  area_difference = 1230 →
  larger_width * larger_length - smaller_width * smaller_length = area_difference →
  smaller_length = 8 :=
by sorry

end smallest_room_length_l2957_295769


namespace vanessa_album_pictures_l2957_295734

/-- The number of albums created by Vanessa -/
def num_albums : ℕ := 10

/-- The number of pictures from the phone in each album -/
def phone_pics_per_album : ℕ := 8

/-- The number of pictures from the camera in each album -/
def camera_pics_per_album : ℕ := 4

/-- The total number of pictures in each album -/
def pics_per_album : ℕ := phone_pics_per_album + camera_pics_per_album

theorem vanessa_album_pictures :
  pics_per_album = 12 :=
sorry

end vanessa_album_pictures_l2957_295734


namespace arithmetic_sequence_general_term_l2957_295798

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 7 = 12)
  (h_product : a 4 * a 5 = 35) :
  (∀ n : ℕ, a n = 2 * n - 3) ∨ (∀ n : ℕ, a n = 15 - 2 * n) :=
sorry

end arithmetic_sequence_general_term_l2957_295798


namespace power_of_four_three_halves_l2957_295756

theorem power_of_four_three_halves : (4 : ℝ) ^ (3/2) = 8 := by
  sorry

end power_of_four_three_halves_l2957_295756


namespace no_solution_exists_l2957_295729

theorem no_solution_exists : ¬∃ (f c₁ c₂ : ℕ), 
  (f > 0) ∧ (c₁ > 0) ∧ (c₂ > 0) ∧ 
  (∃ k : ℕ, f = k * (c₁ + c₂)) ∧
  (f + 5 = 2 * ((c₁ + 5) + (c₂ + 5))) :=
by sorry

end no_solution_exists_l2957_295729


namespace distinct_triangles_count_l2957_295763

/-- Represents a triangle with sides divided into segments -/
structure DividedTriangle where
  sides : ℕ  -- number of segments each side is divided into

/-- Counts the number of distinct triangles formed from division points -/
def count_distinct_triangles (t : DividedTriangle) : ℕ :=
  let total_points := (t.sides - 1) * 3
  let total_triangles := (total_points.choose 3)
  let parallel_sided := 3 * (t.sides - 1)^2
  let double_parallel := 3 * (t.sides - 1)
  let triple_parallel := 1
  total_triangles - parallel_sided + double_parallel - triple_parallel

/-- The main theorem stating the number of distinct triangles -/
theorem distinct_triangles_count (t : DividedTriangle) (h : t.sides = 8) :
  count_distinct_triangles t = 216 := by
  sorry

#eval count_distinct_triangles ⟨8⟩

end distinct_triangles_count_l2957_295763


namespace rounded_number_accuracy_l2957_295779

/-- Represents a number with its value and accuracy -/
structure ApproximateNumber where
  value : ℝ
  accuracy : ℕ

/-- Defines the concept of "accurate to the hundreds place" -/
def accurate_to_hundreds (n : ApproximateNumber) : Prop :=
  ∃ (k : ℤ), n.value = (k * 100 : ℝ) ∧ 
  ∀ (m : ℤ), |n.value - (m * 100 : ℝ)| ≥ 50

/-- The main theorem to prove -/
theorem rounded_number_accuracy :
  let n := ApproximateNumber.mk (8.80 * 10^4) 2
  accurate_to_hundreds n :=
by sorry

end rounded_number_accuracy_l2957_295779


namespace five_ruble_coins_l2957_295754

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  one : Nat
  two : Nat
  five : Nat
  ten : Nat

/-- The problem setup -/
def coin_problem (c : CoinCounts) : Prop :=
  c.one + c.two + c.five + c.ten = 25 ∧
  c.one + c.five + c.ten = 19 ∧
  c.one + c.two + c.five = 20 ∧
  c.two + c.five + c.ten = 16

/-- The theorem to be proved -/
theorem five_ruble_coins (c : CoinCounts) : 
  coin_problem c → c.five = 5 := by
  sorry

end five_ruble_coins_l2957_295754


namespace equal_area_rectangles_l2957_295720

/-- Proves that given two rectangles with equal area, where one rectangle measures 8 inches by 15 inches
    and the other is 30 inches wide, the length of the second rectangle is 4 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_width : ℝ) 
    (h1 : carol_length = 8)
    (h2 : carol_width = 15)
    (h3 : jordan_width = 30)
    (h4 : carol_length * carol_width = jordan_width * jordan_length) :
    jordan_length = 4 :=
  sorry

end equal_area_rectangles_l2957_295720


namespace sequence_solution_l2957_295748

def x (n : ℕ+) : ℚ := n / (n + 2016)

theorem sequence_solution :
  ∃ (m n : ℕ+), x 2016 = x m * x n ∧ m = 4032 ∧ n = 6048 :=
by sorry

end sequence_solution_l2957_295748


namespace garage_sale_items_count_l2957_295770

theorem garage_sale_items_count 
  (prices : Finset ℕ) 
  (radio_price : ℕ) 
  (h_distinct : prices.card = prices.toList.length)
  (h_ninth_highest : (prices.filter (· > radio_price)).card = 8)
  (h_thirty_fifth_lowest : (prices.filter (· < radio_price)).card = 34)
  (h_radio_in_prices : radio_price ∈ prices) :
  prices.card = 43 := by
sorry

end garage_sale_items_count_l2957_295770


namespace parking_arrangements_l2957_295738

theorem parking_arrangements (total_spaces : ℕ) (cars : ℕ) (consecutive_empty : ℕ) 
  (h1 : total_spaces = 12) 
  (h2 : cars = 8) 
  (h3 : consecutive_empty = 4) : 
  (Nat.factorial cars) * (total_spaces - cars - consecutive_empty + 1) = 362880 := by
  sorry

end parking_arrangements_l2957_295738


namespace mike_total_hours_l2957_295703

/-- Calculate the total hours worked given hours per day and number of days -/
def total_hours (hours_per_day : ℕ) (days : ℕ) : ℕ := hours_per_day * days

/-- Proof that Mike worked 15 hours in total -/
theorem mike_total_hours : total_hours 3 5 = 15 := by
  sorry

end mike_total_hours_l2957_295703


namespace x_greater_than_half_l2957_295782

theorem x_greater_than_half (x : ℝ) (h : (1/2) * x = 1) : 
  (x - 1/2) / (1/2) * 100 = 300 := by
  sorry

end x_greater_than_half_l2957_295782


namespace empty_graph_l2957_295753

theorem empty_graph (x y : ℝ) : ¬∃ (x y : ℝ), 3*x^2 + y^2 - 9*x - 4*y + 17 = 0 := by
  sorry

end empty_graph_l2957_295753


namespace intersection_inequality_solution_l2957_295788

/-- Given two linear functions y₁ = ax + b and y₂ = cx + d with a > c > 0,
    intersecting at the point (2, m), prove that the solution set of
    the inequality (a-c)x ≤ d-b is x ≤ 2. -/
theorem intersection_inequality_solution
  (a b c d m : ℝ)
  (h1 : a > c)
  (h2 : c > 0)
  (h3 : a * 2 + b = c * 2 + d)
  (h4 : a * 2 + b = m) :
  ∀ x, (a - c) * x ≤ d - b ↔ x ≤ 2 := by
  sorry

end intersection_inequality_solution_l2957_295788


namespace max_third_altitude_is_seven_l2957_295702

/-- A scalene triangle with two known altitudes -/
structure ScaleneTriangle where
  /-- The length of the first known altitude -/
  altitude1 : ℝ
  /-- The length of the second known altitude -/
  altitude2 : ℝ
  /-- The triangle is scalene -/
  scalene : True
  /-- The known altitudes have lengths 5 and 15 -/
  altitudes_given : altitude1 = 5 ∧ altitude2 = 15

/-- The maximum possible integer length of the third altitude -/
def max_third_altitude (triangle : ScaleneTriangle) : ℕ :=
  7

/-- Theorem stating that the maximum possible integer length of the third altitude is 7 -/
theorem max_third_altitude_is_seven (triangle : ScaleneTriangle) :
  max_third_altitude triangle = 7 := by
  sorry

end max_third_altitude_is_seven_l2957_295702


namespace trees_in_yard_l2957_295791

/-- Given a yard of length 275 meters with trees planted at equal distances,
    one tree at each end, and 11 meters between consecutive trees,
    prove that there are 26 trees in total. -/
theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) : 
  yard_length = 275 → 
  tree_distance = 11 → 
  (yard_length - tree_distance) % tree_distance = 0 →
  (yard_length - tree_distance) / tree_distance + 2 = 26 := by
  sorry

end trees_in_yard_l2957_295791


namespace excavation_time_equality_l2957_295789

/-- Represents the dimensions of an excavation site -/
structure Dimensions where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of an excavation site given its dimensions -/
def volume (d : Dimensions) : ℝ := d.depth * d.length * d.breadth

/-- The number of days required to dig an excavation site is directly proportional to its volume when the number of laborers is constant -/
axiom days_proportional_to_volume {d1 d2 : Dimensions} {days1 : ℝ} (h : volume d1 = volume d2) :
  days1 = days1 * (volume d2 / volume d1)

theorem excavation_time_equality (initial : Dimensions) (new : Dimensions) (initial_days : ℝ) 
    (h_initial : initial = { depth := 100, length := 25, breadth := 30 })
    (h_new : new = { depth := 75, length := 20, breadth := 50 })
    (h_initial_days : initial_days = 12) :
    initial_days = initial_days * (volume new / volume initial) := by
  sorry

end excavation_time_equality_l2957_295789


namespace jess_remaining_distance_l2957_295776

/-- The remaining distance Jess must walk to arrive at work -/
def remaining_distance (store_distance gallery_distance work_distance walked_distance : ℕ) : ℕ :=
  store_distance + gallery_distance + work_distance - walked_distance

/-- Proof that Jess must walk 20 more blocks to arrive at work -/
theorem jess_remaining_distance :
  remaining_distance 11 6 8 5 = 20 := by
  sorry

end jess_remaining_distance_l2957_295776


namespace greatest_n_value_l2957_295706

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 3600) : n ≤ 5 ∧ ∃ (m : ℤ), m > 5 → 101 * m^2 > 3600 := by
  sorry

end greatest_n_value_l2957_295706


namespace triangle_cosine_inequality_l2957_295773

theorem triangle_cosine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) :
  1/3 * (Real.cos A + Real.cos B + Real.cos C) ≤ 1/2 ∧
  1/2 ≤ Real.sqrt (1/3 * (Real.cos A^2 + Real.cos B^2 + Real.cos C^2)) := by
  sorry

end triangle_cosine_inequality_l2957_295773


namespace divisibility_property_l2957_295797

theorem divisibility_property (n : ℕ) : n ≥ 1 ∧ n ∣ (3^n + 1) ∧ n ∣ (11^n + 1) ↔ n = 1 ∨ n = 2 := by
  sorry

end divisibility_property_l2957_295797


namespace bookstore_location_l2957_295740

/-- The floor number of the academy -/
def academy_floor : ℕ := 7

/-- The number of floors the reading room is above the academy -/
def reading_room_above_academy : ℕ := 4

/-- The number of floors the bookstore is below the reading room -/
def bookstore_below_reading_room : ℕ := 9

/-- The floor number of the bookstore -/
def bookstore_floor : ℕ := academy_floor + reading_room_above_academy - bookstore_below_reading_room

theorem bookstore_location : bookstore_floor = 2 := by
  sorry

end bookstore_location_l2957_295740
