import Mathlib

namespace functional_equation_solution_l19_1989

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧
  ∀ x y : ℝ, (x - y) * (f (f (x^2)) - f (f (y^2))) = (f x + f y) * (f x - f y)^2

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = cx for some constant c -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end functional_equation_solution_l19_1989


namespace sum_of_reciprocal_roots_l19_1928

theorem sum_of_reciprocal_roots (x₁ x₂ : ℝ) : 
  x₁^2 + 2*x₁ - 3 = 0 → x₂^2 + 2*x₂ - 3 = 0 → x₁ ≠ x₂ → 
  (1/x₁ + 1/x₂ : ℝ) = 2/3 := by sorry

end sum_of_reciprocal_roots_l19_1928


namespace solution_theorem_l19_1931

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (z : ℂ) : Prop := (1 - i)^2 * z = 3 + 2*i

-- State the theorem
theorem solution_theorem :
  ∃ (z : ℂ), given_equation z ∧ z = -1 + (3/2) * i :=
sorry

end solution_theorem_l19_1931


namespace converse_of_quadratic_eq_l19_1992

theorem converse_of_quadratic_eq (x : ℝ) : x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0 := by sorry

end converse_of_quadratic_eq_l19_1992


namespace largest_811_double_l19_1981

/-- Converts a number from base 8 to base 10 --/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 --/
def base10To8 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 11 --/
def base10To11 (n : ℕ) : ℕ := sorry

/-- Checks if a number is an 8-11 double --/
def is811Double (n : ℕ) : Prop :=
  base10To11 (base8To10 (base10To8 n)) = 2 * n

/-- The largest 8-11 double is 504 --/
theorem largest_811_double :
  (∀ m : ℕ, m > 504 → ¬ is811Double m) ∧ is811Double 504 := by sorry

end largest_811_double_l19_1981


namespace twelve_hour_clock_chimes_90_l19_1977

/-- Calculates the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a clock that chimes on the hour and half-hour -/
structure ChimingClock where
  hours : ℕ
  chimes_on_hour : ℕ → ℕ
  chimes_on_half_hour : ℕ

/-- Calculates the total number of chimes for a ChimingClock over its set hours -/
def total_chimes (clock : ChimingClock) : ℕ :=
  sum_to_n clock.hours + clock.hours * clock.chimes_on_half_hour

/-- Theorem stating that a clock chiming the hour count on the hour and once on the half-hour,
    over 12 hours, will chime 90 times in total -/
theorem twelve_hour_clock_chimes_90 :
  ∃ (clock : ChimingClock),
    clock.hours = 12 ∧
    clock.chimes_on_hour = id ∧
    clock.chimes_on_half_hour = 1 ∧
    total_chimes clock = 90 := by
  sorry

end twelve_hour_clock_chimes_90_l19_1977


namespace smallest_expression_l19_1959

theorem smallest_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) ≤ min ((a + b) / 2) (min (Real.sqrt (a * b)) (Real.sqrt ((a^2 + b^2) / 2))) :=
sorry

end smallest_expression_l19_1959


namespace passing_percentage_l19_1955

/-- Given a total of 500 marks, a student who got 150 marks and failed by 50 marks,
    prove that the percentage needed to pass is 40%. -/
theorem passing_percentage (total_marks : ℕ) (obtained_marks : ℕ) (failing_margin : ℕ) :
  total_marks = 500 →
  obtained_marks = 150 →
  failing_margin = 50 →
  (obtained_marks + failing_margin) / total_marks * 100 = 40 := by
  sorry


end passing_percentage_l19_1955


namespace hendrix_class_size_l19_1985

theorem hendrix_class_size :
  ∀ (initial_students : ℕ),
    (initial_students + 20 : ℚ) * (2/3) = 120 →
    initial_students = 160 := by
  sorry

end hendrix_class_size_l19_1985


namespace quadratic_function_properties_l19_1994

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ f (-1)) ∧  -- f(-1) is the minimum value
  f (-1) = -3 ∧          -- f(-1) = -3
  f 1 = 5                -- f(1) = 5
  := by sorry

end quadratic_function_properties_l19_1994


namespace jasmine_cut_length_l19_1963

def ribbon_length : ℕ := 10
def janice_cut_length : ℕ := 2

theorem jasmine_cut_length :
  ∀ (jasmine_cut : ℕ),
    jasmine_cut ≠ janice_cut_length →
    ribbon_length % jasmine_cut = 0 →
    ribbon_length % janice_cut_length = 0 →
    jasmine_cut = 5 :=
by sorry

end jasmine_cut_length_l19_1963


namespace equidistant_function_l19_1948

def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * Complex.I) * z

theorem equidistant_function (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_equidistant : ∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z - 1))
  (h_norm : Complex.abs (a + b * Complex.I) = 5) :
  b^2 = 99/4 := by
sorry

end equidistant_function_l19_1948


namespace unique_integer_solution_l19_1952

theorem unique_integer_solution (m : ℤ) : 
  (∃! (x : ℤ), |2*x - m| ≤ 1) ∧ 
  (∀ (x : ℤ), |2*x - m| ≤ 1 → x = 2) → 
  m = 4 :=
by sorry

end unique_integer_solution_l19_1952


namespace gear_speed_ratio_l19_1903

structure Gear where
  teeth : ℕ
  speed : ℚ

def meshed (g1 g2 : Gear) : Prop :=
  g1.teeth * g1.speed = g2.teeth * g2.speed

theorem gear_speed_ratio 
  (A B C D : Gear)
  (h_mesh_AB : meshed A B)
  (h_mesh_BC : meshed B C)
  (h_mesh_CD : meshed C D)
  (h_prime_p : Nat.Prime A.teeth)
  (h_prime_q : Nat.Prime B.teeth)
  (h_prime_r : Nat.Prime C.teeth)
  (h_prime_s : Nat.Prime D.teeth)
  (h_distinct : A.teeth ≠ B.teeth ∧ A.teeth ≠ C.teeth ∧ A.teeth ≠ D.teeth ∧
                B.teeth ≠ C.teeth ∧ B.teeth ≠ D.teeth ∧ C.teeth ≠ D.teeth)
  (h_speed_ratio : A.speed / A.teeth = B.speed / B.teeth ∧
                   B.speed / B.teeth = C.speed / C.teeth ∧
                   C.speed / C.teeth = D.speed / D.teeth) :
  ∃ (k : ℚ), A.speed = k * D.teeth * C.teeth ∧
             B.speed = k * D.teeth * A.teeth ∧
             C.speed = k * D.teeth * C.teeth ∧
             D.speed = k * C.teeth * A.teeth := by
  sorry

end gear_speed_ratio_l19_1903


namespace complex_equation_solution_l19_1926

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l19_1926


namespace not_always_equal_l19_1983

theorem not_always_equal (a b c : ℝ) (h1 : a = b - c) :
  (a - b/2)^2 = (c - b/2)^2 → a = c ∨ a + c = b := by sorry

end not_always_equal_l19_1983


namespace age_sum_proof_l19_1929

theorem age_sum_proof (child_age mother_age : ℕ) : 
  child_age = 10 →
  mother_age = 3 * child_age →
  child_age + mother_age = 40 := by
sorry

end age_sum_proof_l19_1929


namespace marias_profit_is_75_l19_1915

/-- Calculates Maria's profit from bread sales given the specified conditions. -/
def marias_profit (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (afternoon_price_ratio : ℚ) (evening_price : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3 * morning_price
  let afternoon_loaves := total_loaves - total_loaves / 3
  let afternoon_sales := afternoon_loaves / 2 * (afternoon_price_ratio * morning_price)
  let evening_loaves := afternoon_loaves - afternoon_loaves / 2
  let evening_sales := evening_loaves * evening_price
  let total_revenue := morning_sales + afternoon_sales + evening_sales
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

/-- Theorem stating that Maria's profit is $75 given the specified conditions. -/
theorem marias_profit_is_75 : 
  marias_profit 60 1 3 (3/4) (3/2) = 75 := by
  sorry

end marias_profit_is_75_l19_1915


namespace largest_multiple_seven_l19_1914

theorem largest_multiple_seven (n : ℤ) : n = 147 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -150) ∧ 
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) → (-m > -150) → m ≤ n) := by
sorry

end largest_multiple_seven_l19_1914


namespace simplify_expression_1_simplify_expression_2_l19_1998

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 
  a^2 - 3*a + 1 - a^2 + 6*a - 7 = 3*a - 6 := by sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℝ) : 
  (3*m^2*n - 5*m*n) - 3*(4*m^2*n - 5*m*n) = -9*m^2*n + 10*m*n := by sorry

end simplify_expression_1_simplify_expression_2_l19_1998


namespace intersection_of_M_and_N_l19_1918

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem intersection_of_M_and_N : M ∩ N = {3, 4} := by sorry

end intersection_of_M_and_N_l19_1918


namespace common_solution_proof_l19_1949

theorem common_solution_proof :
  let x : ℝ := Real.rpow 2 (1/3) - 2
  let y : ℝ := Real.rpow 2 (2/3)
  (y = (x + 2)^2) ∧ (x * y + y = 2) := by
  sorry

end common_solution_proof_l19_1949


namespace league_games_l19_1930

theorem league_games (n : ℕ) (h : n = 10) : (n.choose 2) = 45 := by
  sorry

end league_games_l19_1930


namespace part_one_part_two_l19_1976

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  (b / a) = b - a

/-- Part 1: Prove that 3x = 4.5 is a difference solution equation -/
theorem part_one : is_difference_solution_equation 3 4.5 := by sorry

/-- Part 2: Prove that 5x - m = 1 is a difference solution equation when m = 21/4 -/
theorem part_two : is_difference_solution_equation 5 ((21/4) + 1) := by sorry

end part_one_part_two_l19_1976


namespace ten_caterpillars_left_l19_1990

/-- The number of caterpillars left on a tree after some changes -/
def caterpillars_left (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

/-- Theorem: Given the initial conditions, prove that 10 caterpillars are left on the tree -/
theorem ten_caterpillars_left : 
  caterpillars_left 14 4 8 = 10 := by
  sorry

end ten_caterpillars_left_l19_1990


namespace triangle_side_length_l19_1917

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 10 →
  a / Real.sin A = b / Real.sin B →
  b = 5 * Real.sqrt 6 := by
sorry

end triangle_side_length_l19_1917


namespace bill_experience_l19_1974

/-- Represents the work experience of a person -/
structure Experience where
  current : ℕ
  fiveYearsAgo : ℕ

/-- The problem setup -/
def libraryProblem : Prop := ∃ (bill joan : Experience),
  -- Bill's current age
  40 = bill.current + bill.fiveYearsAgo
  -- Joan's current age
  ∧ 50 = joan.current + joan.fiveYearsAgo
  -- 5 years ago, Joan had 3 times as much experience as Bill
  ∧ joan.fiveYearsAgo = 3 * bill.fiveYearsAgo
  -- Now, Joan has twice as much experience as Bill
  ∧ joan.current = 2 * bill.current
  -- Bill's current experience is 10 years
  ∧ bill.current = 10

/-- The theorem to prove -/
theorem bill_experience : libraryProblem := by sorry

end bill_experience_l19_1974


namespace chord_squares_difference_l19_1996

/-- Given a circle with a chord at distance h from the center, and two squares inscribed in the 
segments subtended by the chord (with two adjacent vertices on the arc and two on the chord or 
its extension), the difference in the side lengths of these squares is 8h/5. -/
theorem chord_squares_difference (h : ℝ) (h_pos : h > 0) : ℝ := by
  sorry

end chord_squares_difference_l19_1996


namespace exists_finite_harmonic_progression_no_infinite_harmonic_progression_l19_1932

/-- A sequence of positive integers is in harmonic progression if their reciprocals form an arithmetic progression. -/
def IsHarmonicProgression (s : ℕ → ℕ) : Prop :=
  ∃ d : ℚ, ∀ i j : ℕ, (1 : ℚ) / s i - (1 : ℚ) / s j = d * (i - j)

/-- For any natural number N, there exists a strictly increasing sequence of N positive integers in harmonic progression. -/
theorem exists_finite_harmonic_progression (N : ℕ) :
    ∃ (s : ℕ → ℕ), (∀ i < N, s i < s (i + 1)) ∧ IsHarmonicProgression s :=
  sorry

/-- There does not exist a strictly increasing infinite sequence of positive integers in harmonic progression. -/
theorem no_infinite_harmonic_progression :
    ¬∃ (s : ℕ → ℕ), (∀ i : ℕ, s i < s (i + 1)) ∧ IsHarmonicProgression s :=
  sorry

end exists_finite_harmonic_progression_no_infinite_harmonic_progression_l19_1932


namespace unique_solution_system_l19_1916

theorem unique_solution_system (x y z t : ℝ) :
  (x * y + z + t = 1) ∧
  (y * z + t + x = 3) ∧
  (z * t + x + y = -1) ∧
  (t * x + y + z = 1) →
  (x = 1 ∧ y = 0 ∧ z = -1 ∧ t = 2) :=
by sorry

end unique_solution_system_l19_1916


namespace sum_of_coefficients_is_one_l19_1925

theorem sum_of_coefficients_is_one : 
  let p (x : ℝ) := 3*(x^8 - 2*x^5 + 4*x^3 - 6) - 5*(2*x^4 + 3*x - 7) + 6*(x^6 - x^2 + 1)
  p 1 = 1 := by sorry

end sum_of_coefficients_is_one_l19_1925


namespace negation_of_forall_even_square_plus_n_l19_1951

theorem negation_of_forall_even_square_plus_n :
  (¬ ∀ n : ℕ, Even (n^2 + n)) ↔ (∃ n : ℕ, ¬ Even (n^2 + n)) := by
  sorry

end negation_of_forall_even_square_plus_n_l19_1951


namespace no_roots_implies_negative_a_l19_1927

theorem no_roots_implies_negative_a :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 := by
  sorry

end no_roots_implies_negative_a_l19_1927


namespace jasmine_solution_percentage_l19_1908

theorem jasmine_solution_percentage
  (initial_volume : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 80)
  (h2 : added_jasmine = 8)
  (h3 : added_water = 12)
  (h4 : final_percentage = 16)
  : (initial_volume * (final_percentage / 100) - added_jasmine) / initial_volume * 100 = 10 :=
by
  sorry

end jasmine_solution_percentage_l19_1908


namespace downstream_distance_l19_1988

/-- Prove that given the conditions of a boat rowing upstream and downstream,
    the distance rowed downstream is 200 km. -/
theorem downstream_distance
  (boat_speed : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (downstream_time : ℝ)
  (h1 : boat_speed = 14)
  (h2 : upstream_distance = 96)
  (h3 : upstream_time = 12)
  (h4 : downstream_time = 10)
  (h5 : upstream_distance / upstream_time = boat_speed - (boat_speed - upstream_distance / upstream_time)) :
  (boat_speed + (boat_speed - upstream_distance / upstream_time)) * downstream_time = 200 := by
sorry

end downstream_distance_l19_1988


namespace solution_set_when_m_is_2_range_of_m_for_inequality_l19_1921

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - 2*m| - |x + m|

-- Part 1
theorem solution_set_when_m_is_2 :
  let m : ℝ := 2
  ∀ x : ℝ, f m x ≥ 1 ↔ -2 < x ∧ x ≤ 1/2 :=
sorry

-- Part 2
theorem range_of_m_for_inequality :
  ∀ m : ℝ, m > 0 →
  (∀ x t : ℝ, f m x ≤ |t + 3| + |t - 2|) ↔
  (0 < m ∧ m ≤ 5/3) :=
sorry

end solution_set_when_m_is_2_range_of_m_for_inequality_l19_1921


namespace M_equals_N_l19_1902

def M : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k - 1) * Real.pi}

theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l19_1902


namespace positive_number_and_square_sum_l19_1954

theorem positive_number_and_square_sum : ∃ (n : ℝ), n > 0 ∧ n^2 + n = 210 ∧ n = 14 ∧ n^3 = 2744 := by
  sorry

end positive_number_and_square_sum_l19_1954


namespace simplify_fraction_l19_1905

theorem simplify_fraction : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end simplify_fraction_l19_1905


namespace image_of_3_4_preimage_of_1_neg6_l19_1999

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (3, 4)
theorem image_of_3_4 : f (3, 4) = (7, 12) := by sorry

-- Definition of preimage
def preimage (f : ℝ × ℝ → ℝ × ℝ) (y : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {x | f x = y}

-- Theorem for the preimage of (1, -6)
theorem preimage_of_1_neg6 : preimage f (1, -6) = {(-2, 3), (3, -2)} := by sorry

end image_of_3_4_preimage_of_1_neg6_l19_1999


namespace yellow_marble_probability_l19_1933

/-- Represents the composition of a bag of marbles -/
structure BagComposition where
  color1 : ℕ
  color2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def drawProbability (bag : BagComposition) (colorCount : ℕ) : ℚ :=
  colorCount / (bag.color1 + bag.color2)

/-- The main theorem statement -/
theorem yellow_marble_probability
  (bagX : BagComposition)
  (bagY : BagComposition)
  (bagZ : BagComposition)
  (hX : bagX = ⟨5, 3⟩)
  (hY : bagY = ⟨8, 2⟩)
  (hZ : bagZ = ⟨3, 4⟩) :
  let probWhiteX := drawProbability bagX bagX.color1
  let probYellowY := drawProbability bagY bagY.color1
  let probBlackX := drawProbability bagX bagX.color2
  let probYellowZ := drawProbability bagZ bagZ.color1
  probWhiteX * probYellowY + probBlackX * probYellowZ = 37 / 56 :=
sorry

end yellow_marble_probability_l19_1933


namespace sara_initial_savings_l19_1944

/-- Sara's initial savings -/
def S : ℕ := sorry

/-- Number of weeks -/
def weeks : ℕ := 820

/-- Sara's weekly savings -/
def sara_weekly : ℕ := 10

/-- Jim's weekly savings -/
def jim_weekly : ℕ := 15

/-- Theorem stating that Sara's initial savings is $4100 -/
theorem sara_initial_savings :
  S = 4100 ∧
  S + sara_weekly * weeks = jim_weekly * weeks :=
sorry

end sara_initial_savings_l19_1944


namespace seven_at_eight_equals_nineteen_thirds_l19_1972

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := (5 * a - 2 * b) / 3

-- Theorem statement
theorem seven_at_eight_equals_nineteen_thirds :
  at_op 7 8 = 19 / 3 := by sorry

end seven_at_eight_equals_nineteen_thirds_l19_1972


namespace monomial_type_sum_l19_1958

/-- Two monomials are of the same type if they have the same variables raised to the same powers -/
def same_type_monomials (m n : ℕ) : Prop :=
  m - 1 = 2 ∧ n + 1 = 2

theorem monomial_type_sum (m n : ℕ) :
  same_type_monomials m n → m + n = 4 := by
  sorry

end monomial_type_sum_l19_1958


namespace hawkeye_remaining_money_l19_1975

/-- Calculates the remaining money after battery charges. -/
def remaining_money (cost_per_charge : ℚ) (num_charges : ℕ) (budget : ℚ) : ℚ :=
  budget - (cost_per_charge * num_charges)

/-- Theorem: Given the specified conditions, the remaining money is $6. -/
theorem hawkeye_remaining_money :
  remaining_money (35/10) 4 20 = 6 := by
  sorry

end hawkeye_remaining_money_l19_1975


namespace factorization_equality_l19_1980

theorem factorization_equality (x : ℝ) :
  (4 * x^3 + 100 * x^2 - 28) - (-9 * x^3 + 2 * x^2 - 28) = 13 * x^2 * (x + 7) := by
  sorry

end factorization_equality_l19_1980


namespace orange_orchard_composition_l19_1965

/-- Represents an orange orchard with flat and hilly areas. -/
structure Orchard :=
  (total_acres : ℕ)
  (sampled_acres : ℕ)
  (flat_sampled : ℕ)
  (hilly_sampled : ℕ)

/-- Checks if the sampling method is valid for the given orchard. -/
def valid_sampling (o : Orchard) : Prop :=
  o.hilly_sampled = 2 * o.flat_sampled + 1 ∧
  o.flat_sampled + o.hilly_sampled = o.sampled_acres

/-- Calculates the number of flat acres in the orchard based on the sampling. -/
def flat_acres (o : Orchard) : ℕ :=
  o.flat_sampled * (o.total_acres / o.sampled_acres)

/-- Calculates the number of hilly acres in the orchard based on the sampling. -/
def hilly_acres (o : Orchard) : ℕ :=
  o.hilly_sampled * (o.total_acres / o.sampled_acres)

/-- Theorem stating the composition of the orange orchard. -/
theorem orange_orchard_composition (o : Orchard) 
  (h1 : o.total_acres = 120)
  (h2 : o.sampled_acres = 10)
  (h3 : valid_sampling o) :
  flat_acres o = 36 ∧ hilly_acres o = 84 :=
sorry

end orange_orchard_composition_l19_1965


namespace trig_problem_l19_1934

theorem trig_problem (α β : ℝ) 
  (h1 : Real.sin α - Real.sin β = -1/3)
  (h2 : Real.cos α - Real.cos β = 1/2)
  (h3 : Real.tan (α + β) = 2/5)
  (h4 : Real.tan (β - π/4) = 1/4) :
  Real.cos (α - β) = 59/72 ∧ Real.tan (α + π/4) = 3/22 := by
  sorry

end trig_problem_l19_1934


namespace white_squares_95th_figure_l19_1912

/-- The number of white squares in the nth figure of the sequence -/
def white_squares (n : ℕ) : ℕ := 8 + 5 * (n - 1)

/-- Theorem: The 95th figure in the sequence has 478 white squares -/
theorem white_squares_95th_figure : white_squares 95 = 478 := by
  sorry

end white_squares_95th_figure_l19_1912


namespace hyperbola_vertices_distance_l19_1907

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 49 = 1

-- Define the distance between vertices
def distance_between_vertices : ℝ := 24

-- Theorem statement
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ, hyperbola_equation x y →
  distance_between_vertices = 24 :=
by sorry

end hyperbola_vertices_distance_l19_1907


namespace a_range_l19_1938

theorem a_range (a : ℝ) : 
  (a + 1)^(-1/4 : ℝ) < (3 - 2*a)^(-1/4 : ℝ) → 2/3 < a ∧ a < 3/2 := by
sorry

end a_range_l19_1938


namespace functional_equation_solution_l19_1904

/-- A function from nonzero reals to nonzero reals -/
def NonzeroRealFunction := ℝ* → ℝ*

/-- The property that f(x+y)(f(x) + f(y)) = f(x)f(y) for all nonzero real x and y -/
def SatisfiesProperty (f : NonzeroRealFunction) : Prop :=
  ∀ x y : ℝ*, f (x + y) * (f x + f y) = f x * f y

/-- The property that a function is increasing -/
def IsIncreasing (f : NonzeroRealFunction) : Prop :=
  ∀ x y : ℝ*, x < y → f x < f y

theorem functional_equation_solution :
  ∀ f : NonzeroRealFunction, IsIncreasing f → SatisfiesProperty f →
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ*, f x = 1 / (a * x) :=
sorry

end functional_equation_solution_l19_1904


namespace tshirt_production_rate_l19_1900

theorem tshirt_production_rate (rate1 : ℝ) (total : ℕ) (rate2 : ℝ) : 
  rate1 = 12 → total = 15 → rate2 = (120 : ℝ) / ((total : ℝ) - 60 / rate1) → rate2 = 6 := by
  sorry

end tshirt_production_rate_l19_1900


namespace negative_representation_is_spending_l19_1901

-- Define a type for monetary transactions
inductive MonetaryTransaction
| Receive (amount : ℤ)
| Spend (amount : ℤ)

-- Define a function to represent transactions as integers
def represent (t : MonetaryTransaction) : ℤ :=
  match t with
  | MonetaryTransaction.Receive amount => amount
  | MonetaryTransaction.Spend amount => -amount

-- State the theorem
theorem negative_representation_is_spending :
  (represent (MonetaryTransaction.Receive 100) = 100) →
  (represent (MonetaryTransaction.Spend 80) = -80) :=
by sorry

end negative_representation_is_spending_l19_1901


namespace chair_cost_l19_1970

theorem chair_cost (total_cost : ℝ) (table_cost : ℝ) (num_chairs : ℕ) :
  total_cost = 135 →
  table_cost = 55 →
  num_chairs = 4 →
  ∃ (chair_cost : ℝ),
    chair_cost * num_chairs = total_cost - table_cost ∧
    chair_cost = 20 :=
by sorry

end chair_cost_l19_1970


namespace perfect_cube_pair_l19_1953

theorem perfect_cube_pair (a b : ℕ+) :
  (∃ (m n : ℕ+), a^3 + 6*a*b + 1 = m^3 ∧ b^3 + 6*a*b + 1 = n^3) →
  a = 1 ∧ b = 1 := by
sorry

end perfect_cube_pair_l19_1953


namespace completing_square_transformation_l19_1941

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end completing_square_transformation_l19_1941


namespace trapezoid_segment_length_squared_l19_1937

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ

/-- The conditions of the trapezoid as described in the problem -/
def trapezoid_conditions (t : Trapezoid) : Prop :=
  -- The longer base is 150 units longer than the shorter base
  ∃ (longer_base : ℝ), longer_base = t.shorter_base + 150
  -- The midline divides the trapezoid into regions with area ratio 3:4
  ∧ (t.shorter_base + t.shorter_base + 75) / (t.shorter_base + 75 + t.shorter_base + 150) = 3 / 4
  -- t.equal_area_segment divides the trapezoid into two equal-area regions
  ∧ ∃ (h₁ : ℝ), 2 * (1/2 * h₁ * (t.shorter_base + t.equal_area_segment)) = 
                 1/2 * t.height * (t.shorter_base + t.shorter_base + 150)

/-- The theorem to be proved -/
theorem trapezoid_segment_length_squared (t : Trapezoid) 
  (h : trapezoid_conditions t) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 300 := by
  sorry

end trapezoid_segment_length_squared_l19_1937


namespace alcohol_mixture_percentage_l19_1973

theorem alcohol_mixture_percentage 
  (initial_volume : ℝ) 
  (initial_percentage : ℝ) 
  (added_alcohol : ℝ) 
  (h1 : initial_volume = 6) 
  (h2 : initial_percentage = 25) 
  (h3 : added_alcohol = 3) : 
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  final_alcohol / final_volume * 100 = 50 := by
sorry


end alcohol_mixture_percentage_l19_1973


namespace abs_negative_thirteen_l19_1909

theorem abs_negative_thirteen : |(-13 : ℤ)| = 13 := by
  sorry

end abs_negative_thirteen_l19_1909


namespace algebraic_expression_value_l19_1962

theorem algebraic_expression_value (y : ℝ) : 
  3 * y^2 - 2 * y + 6 = 8 → (3/2) * y^2 - y + 2 = 3 := by
  sorry

end algebraic_expression_value_l19_1962


namespace velocity_zero_at_two_l19_1924

-- Define the motion equation
def s (t : ℝ) : ℝ := -4 * t^3 + 48 * t

-- Define the velocity function (derivative of s)
def v (t : ℝ) : ℝ := -12 * t^2 + 48

-- Theorem stating that the positive time when velocity is zero is 2
theorem velocity_zero_at_two :
  ∃ (t : ℝ), t > 0 ∧ v t = 0 ∧ t = 2 := by
  sorry

end velocity_zero_at_two_l19_1924


namespace bucket_water_volume_l19_1935

theorem bucket_water_volume (initial_volume : ℝ) (additional_volume : ℝ) : 
  initial_volume = 2 → 
  additional_volume = 460 → 
  (initial_volume * 1000 + additional_volume : ℝ) = 2460 := by
sorry

end bucket_water_volume_l19_1935


namespace certain_fraction_proof_l19_1956

theorem certain_fraction_proof : 
  ∃ (x y : ℚ), (x / y) / (6 / 7) = (7 / 15) / (2 / 3) ∧ x / y = 3 / 5 :=
by sorry

end certain_fraction_proof_l19_1956


namespace coin_flip_probability_l19_1971

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure SixCoins where
  penny : CoinFlip
  nickel : CoinFlip
  dime : CoinFlip
  quarter : CoinFlip
  halfDollar : CoinFlip
  dollar : CoinFlip

/-- The total number of possible outcomes when flipping six coins -/
def totalOutcomes : Nat := 64

/-- Checks if the penny and dime have different outcomes -/
def pennyDimeDifferent (coins : SixCoins) : Prop :=
  coins.penny ≠ coins.dime

/-- Checks if the nickel and quarter have the same outcome -/
def nickelQuarterSame (coins : SixCoins) : Prop :=
  coins.nickel = coins.quarter

/-- Counts the number of favorable outcomes -/
def favorableOutcomes : Nat := 16

/-- The probability of the specified event -/
def probability : ℚ := 1 / 4

theorem coin_flip_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = probability :=
sorry

end coin_flip_probability_l19_1971


namespace min_distance_between_curves_l19_1913

/-- The minimum distance between two points on given curves with the same y-coordinate -/
theorem min_distance_between_curves : ∃ (d : ℝ), d = (5 + Real.log 2) / 4 ∧
  ∀ (x₁ x₂ y : ℝ), 
    y = Real.exp (2 * x₁ + 1) → 
    y = Real.sqrt (2 * x₂ - 1) → 
    d ≤ |x₂ - x₁| := by
  sorry

end min_distance_between_curves_l19_1913


namespace factor_t_squared_minus_144_l19_1993

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end factor_t_squared_minus_144_l19_1993


namespace add_3577_minutes_to_start_time_l19_1979

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startDateTime : DateTime :=
  { year := 2020, month := 12, day := 31, hour := 18, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3577

/-- The resulting DateTime after adding minutes -/
def resultDateTime : DateTime :=
  { year := 2021, month := 1, day := 3, hour := 5, minute := 37 }

theorem add_3577_minutes_to_start_time :
  addMinutes startDateTime minutesToAdd = resultDateTime := by sorry

end add_3577_minutes_to_start_time_l19_1979


namespace vector_subtraction_and_scalar_multiplication_l19_1911

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 2 → ℝ := ![3, -8]
  let v₂ : Fin 2 → ℝ := ![4, 6]
  let scalar : ℝ := -5
  let result : Fin 2 → ℝ := ![23, 22]
  v₁ - scalar • v₂ = result := by sorry

end vector_subtraction_and_scalar_multiplication_l19_1911


namespace floor_function_unique_l19_1986

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def Condition1 (f : RealFunction) : Prop :=
  ∀ x y : ℝ, f x + f y + 1 ≥ f (x + y) ∧ f (x + y) ≥ f x + f y

def Condition2 (f : RealFunction) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f 0 ≥ f x

def Condition3 (f : RealFunction) : Prop :=
  f (-1) = -1 ∧ f 1 = 1

-- Main theorem
theorem floor_function_unique (f : RealFunction)
  (h1 : Condition1 f) (h2 : Condition2 f) (h3 : Condition3 f) :
  ∀ x : ℝ, f x = ⌊x⌋ := by sorry

end floor_function_unique_l19_1986


namespace sum_of_a_and_b_l19_1923

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 40) 
  (eq2 : 3 * a + 4 * b = 38) : 
  a + b = 74 / 7 := by
sorry

end sum_of_a_and_b_l19_1923


namespace exists_spanish_couple_l19_1950

-- Define the set S
def S : Set ℝ := {x | ∃ (a b : ℕ), x = (a - 1) / b}

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the Spanish Couple property
def SpanishCouple (f g : ℝ → ℝ) : Prop :=
  (∀ x ∈ S, f x ∈ S) ∧
  (∀ x ∈ S, g x ∈ S) ∧
  StrictlyIncreasing f ∧
  StrictlyIncreasing g ∧
  ∀ x ∈ S, f (g (g x)) < g (f x)

-- Theorem statement
theorem exists_spanish_couple : ∃ f g, SpanishCouple f g := by
  sorry

end exists_spanish_couple_l19_1950


namespace negation_of_statement_l19_1968

theorem negation_of_statement :
  ¬(∀ x : ℝ, x ≠ 0 → x^2 - 4 > 0) ↔ 
  (∃ x : ℝ, x ≠ 0 ∧ x^2 - 4 ≤ 0) :=
by sorry

end negation_of_statement_l19_1968


namespace random_course_selection_probability_l19_1943

theorem random_course_selection_probability 
  (courses : Finset String) 
  (h1 : courses.card = 4) 
  (h2 : courses.Nonempty) 
  (selected_course : String) 
  (h3 : selected_course ∈ courses) :
  (Finset.filter (· = selected_course) courses).card / courses.card = (1 : ℚ) / 4 :=
sorry

end random_course_selection_probability_l19_1943


namespace hannah_remaining_money_l19_1987

def county_fair_expenses (initial_amount : ℚ) (ride_percentage : ℚ) (game_percentage : ℚ)
  (dessert_cost : ℚ) (cotton_candy_cost : ℚ) (hotdog_cost : ℚ)
  (keychain_cost : ℚ) (poster_cost : ℚ) (attraction_cost : ℚ) : ℚ :=
  let ride_expense := initial_amount * ride_percentage
  let game_expense := initial_amount * game_percentage
  let food_souvenir_expense := dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost + poster_cost + attraction_cost
  initial_amount - (ride_expense + game_expense + food_souvenir_expense)

theorem hannah_remaining_money :
  county_fair_expenses 120 0.4 0.15 8 5 6 7 10 15 = 3 := by
  sorry

end hannah_remaining_money_l19_1987


namespace rhombus_side_length_l19_1978

/-- The side length of a rhombus given its area and diagonal ratio -/
theorem rhombus_side_length (K : ℝ) (h : K > 0) : ∃ (s : ℝ),
  s > 0 ∧
  ∃ (d₁ d₂ : ℝ),
    d₁ > 0 ∧ d₂ > 0 ∧
    d₂ = 3 * d₁ ∧
    K = (1/2) * d₁ * d₂ ∧
    s^2 = (d₁/2)^2 + (d₂/2)^2 ∧
    s = Real.sqrt ((5 * K) / 3) :=
by sorry

end rhombus_side_length_l19_1978


namespace gcd_repeated_digits_l19_1942

def repeat_digit (n : ℕ) : ℕ := n + 1000 * n + 1000000 * n

theorem gcd_repeated_digits :
  ∃ (g : ℕ), g > 0 ∧ 
  (∀ (n : ℕ), 100 ≤ n → n < 1000 → g ∣ repeat_digit n) ∧
  (∀ (d : ℕ), d > 0 → 
    (∀ (n : ℕ), 100 ≤ n → n < 1000 → d ∣ repeat_digit n) → 
    d ∣ g) ∧
  g = 1001001 := by
sorry

#eval 1001001

end gcd_repeated_digits_l19_1942


namespace pascal_ratio_row_l19_1991

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Three consecutive entries in Pascal's Triangle are in ratio 3:4:5 -/
def ratio_condition (n : ℕ) (r : ℕ) : Prop :=
  ∃ (a b c : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a * (pascal n (r+1)) = b * (pascal n r) ∧
    b * (pascal n (r+2)) = c * (pascal n (r+1)) ∧
    3 * b = 4 * a ∧ 4 * c = 5 * b

theorem pascal_ratio_row :
  ∃ (n : ℕ), n = 62 ∧ ∃ (r : ℕ), ratio_condition n r :=
sorry

end pascal_ratio_row_l19_1991


namespace quotient_invariance_problem_solution_l19_1946

theorem quotient_invariance (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / b = (c * a) / (c * b) :=
by sorry

theorem problem_solution : (0.75 : ℝ) / 25 = 7.5 / 250 := by
  have h1 : (0.75 : ℝ) / 25 = (10 * 0.75) / (10 * 25) := by
    apply quotient_invariance 0.75 25 10
    norm_num
    norm_num
  have h2 : (10 * 0.75 : ℝ) = 7.5 := by norm_num
  have h3 : (10 * 25 : ℝ) = 250 := by norm_num
  rw [h1, h2, h3]

end quotient_invariance_problem_solution_l19_1946


namespace freeway_to_traffic_ratio_l19_1940

def total_time : ℝ := 10
def traffic_time : ℝ := 2

theorem freeway_to_traffic_ratio :
  (total_time - traffic_time) / traffic_time = 4 := by
  sorry

end freeway_to_traffic_ratio_l19_1940


namespace product_of_solutions_l19_1960

theorem product_of_solutions (x : ℝ) : 
  (|18 / x - 4| = 3) → (∃ y : ℝ, (|18 / y - 4| = 3) ∧ (x * y = 324 / 7)) :=
by sorry

end product_of_solutions_l19_1960


namespace binomial_coefficient_1300_2_l19_1997

theorem binomial_coefficient_1300_2 : 
  Nat.choose 1300 2 = 844350 := by sorry

end binomial_coefficient_1300_2_l19_1997


namespace smallest_number_with_2020_divisors_l19_1936

def number_of_divisors (n : ℕ) : ℕ :=
  (Nat.factors n).map (· + 1) |>.prod

def is_smallest_with_2020_divisors (n : ℕ) : Prop :=
  number_of_divisors n = 2020 ∧
  ∀ m < n, number_of_divisors m ≠ 2020

theorem smallest_number_with_2020_divisors :
  is_smallest_with_2020_divisors (2^100 * 3^4 * 5 * 7) :=
sorry

end smallest_number_with_2020_divisors_l19_1936


namespace max_q_plus_r_for_1057_l19_1961

theorem max_q_plus_r_for_1057 :
  ∃ (q r : ℕ+), 1057 = 23 * q + r ∧ ∀ (q' r' : ℕ+), 1057 = 23 * q' + r' → q + r ≥ q' + r' :=
by sorry

end max_q_plus_r_for_1057_l19_1961


namespace simple_interest_proof_l19_1945

/-- Given a principal amount where the compound interest for 2 years at 5% per annum is 41,
    prove that the simple interest for the same principal, rate, and time is 40. -/
theorem simple_interest_proof (P : ℝ) : 
  P * (1 + 0.05)^2 - P = 41 → P * 0.05 * 2 = 40 := by
sorry

end simple_interest_proof_l19_1945


namespace vodka_alcohol_consumption_l19_1920

/-- Calculates the amount of pure alcohol consumed by one person when splitting vodka shots. -/
theorem vodka_alcohol_consumption
  (total_shots : ℕ)
  (ounces_per_shot : ℚ)
  (alcohol_percentage : ℚ)
  (h1 : total_shots = 8)
  (h2 : ounces_per_shot = 3/2)
  (h3 : alcohol_percentage = 1/2) :
  (((total_shots : ℚ) / 2) * ounces_per_shot) * alcohol_percentage = 3 := by
  sorry

end vodka_alcohol_consumption_l19_1920


namespace expand_and_simplify_l19_1967

theorem expand_and_simplify (x y : ℝ) : 
  (2*x + 3*y)^2 - 2*x*(2*x - 3*y) = 18*x*y + 9*y^2 := by
  sorry

end expand_and_simplify_l19_1967


namespace polynomial_division_remainder_l19_1922

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 3*X + 1 = (X - 3)^2 * q + (81*X - 161) := by
  sorry

end polynomial_division_remainder_l19_1922


namespace tan_function_property_l19_1939

/-- 
Given a function f(x) = a * tan(b * x) where a and b are positive constants,
if the function has a period of 2π/5 and passes through the point (π/10, 1),
then the product ab equals 5/2.
-/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + 2 * π / 5))) → 
  (a * Real.tan (b * π / 10) = 1) → 
  a * b = 5 / 2 := by
  sorry

end tan_function_property_l19_1939


namespace cuboid_height_calculation_l19_1906

/-- The surface area of a cuboid given its length, breadth, and height. -/
def surfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + l * h + b * h)

/-- Theorem: For a cuboid with surface area 720, length 12, and breadth 6, the height is 16. -/
theorem cuboid_height_calculation (SA l b h : ℝ) 
  (h_SA : SA = 720) 
  (h_l : l = 12) 
  (h_b : b = 6) 
  (h_surface_area : surfaceArea l b h = SA) : h = 16 := by
  sorry

#check cuboid_height_calculation

end cuboid_height_calculation_l19_1906


namespace tree_break_height_l19_1957

/-- Given a tree of height 36 meters that breaks and falls across a road of width 12 meters,
    touching the opposite edge, the height at which the tree broke is 16 meters. -/
theorem tree_break_height :
  ∀ (h : ℝ), 
  h > 0 →
  h < 36 →
  (36 - h)^2 = h^2 + 12^2 →
  h = 16 :=
by sorry

end tree_break_height_l19_1957


namespace peanut_ball_probability_l19_1964

/-- The probability of selecting at least one peanut filling glutinous rice ball -/
theorem peanut_ball_probability : 
  let total_balls : ℕ := 6
  let peanut_balls : ℕ := 2
  let selected_balls : ℕ := 2
  Nat.choose total_balls selected_balls ≠ 0 →
  (Nat.choose peanut_balls selected_balls + 
   peanut_balls * (total_balls - peanut_balls)) / 
  Nat.choose total_balls selected_balls = 3 / 5 :=
by sorry

end peanut_ball_probability_l19_1964


namespace smallest_product_sum_l19_1966

def digits : List Nat := [3, 4, 5, 6, 7]

def is_valid_configuration (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product_sum (a b c d e : Nat) : Nat :=
  (10 * a + b) * (10 * c + d) + e * (10 * a + b)

theorem smallest_product_sum :
  ∀ a b c d e : Nat,
    is_valid_configuration a b c d e →
    product_sum a b c d e ≥ 2448 :=
by sorry

end smallest_product_sum_l19_1966


namespace principal_amount_proof_l19_1910

/-- Proves that the principal amount is 800 given the specified conditions -/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (total_amount : ℝ) : 
  rate = 0.0375 → time = 5 → total_amount = 950 →
  (∃ (principal : ℝ), principal * (1 + rate * time) = total_amount ∧ principal = 800) := by
  sorry

#check principal_amount_proof

end principal_amount_proof_l19_1910


namespace min_sum_with_constraint_l19_1982

theorem min_sum_with_constraint (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y)
  (h : a / x + b / y = 2) :
  x + y ≥ (a + b) / 2 + Real.sqrt (a * b) := by
  sorry

end min_sum_with_constraint_l19_1982


namespace boat_speed_in_still_water_l19_1919

/-- Proves that a boat's speed in still water is 2.5 km/hr given its downstream and upstream travel times -/
theorem boat_speed_in_still_water 
  (distance : ℝ) 
  (downstream_time upstream_time : ℝ) 
  (h1 : distance = 10) 
  (h2 : downstream_time = 3) 
  (h3 : upstream_time = 6) : 
  ∃ (boat_speed : ℝ), boat_speed = 2.5 := by
sorry

end boat_speed_in_still_water_l19_1919


namespace certain_number_problem_l19_1947

theorem certain_number_problem (x y : ℤ) : x = 15 ∧ 2 * x = (y - x) + 19 → y = 26 := by
  sorry

end certain_number_problem_l19_1947


namespace no_solution_absolute_value_equation_l19_1995

theorem no_solution_absolute_value_equation :
  ¬ ∃ x : ℝ, |(-2 * x)| + 6 = 0 := by
  sorry

end no_solution_absolute_value_equation_l19_1995


namespace barbara_butcher_cost_l19_1984

/-- The total cost of Barbara's purchase at the butcher's -/
def total_cost (steak_weight : ℝ) (steak_price : ℝ) (chicken_weight : ℝ) (chicken_price : ℝ) : ℝ :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem: Barbara's total cost at the butcher's is $79.50 -/
theorem barbara_butcher_cost : 
  total_cost 4.5 15 1.5 8 = 79.5 := by
  sorry

#eval total_cost 4.5 15 1.5 8

end barbara_butcher_cost_l19_1984


namespace sqrt_equation_solution_l19_1969

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 3 → x = -1 := by
  sorry

end sqrt_equation_solution_l19_1969
