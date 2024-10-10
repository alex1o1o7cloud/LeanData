import Mathlib

namespace total_study_time_is_135_l1555_155565

def math_time : ℕ := 60

def geography_time : ℕ := math_time / 2

def science_time : ℕ := (math_time + geography_time) / 2

def total_study_time : ℕ := math_time + geography_time + science_time

theorem total_study_time_is_135 : total_study_time = 135 := by
  sorry

end total_study_time_is_135_l1555_155565


namespace unique_solution_l1555_155562

/-- The system of equations and constraint -/
def system (x y z w : ℝ) : Prop :=
  x = Real.sin (z + w + z * w * x) ∧
  y = Real.sin (w + x + w * x * y) ∧
  z = Real.sin (x + y + x * y * z) ∧
  w = Real.sin (y + z + y * z * w) ∧
  Real.cos (x + y + z + w) = 1

/-- There exists exactly one solution to the system -/
theorem unique_solution : ∃! (x y z w : ℝ), system x y z w :=
sorry

end unique_solution_l1555_155562


namespace two_distinct_roots_l1555_155560

theorem two_distinct_roots (p : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ - 3) * (x₁ - 2) - p^2 = 0 ∧ (x₂ - 3) * (x₂ - 2) - p^2 = 0 :=
by sorry

end two_distinct_roots_l1555_155560


namespace quadratic_rewrite_l1555_155538

theorem quadratic_rewrite (x : ℝ) :
  ∃ (b c : ℝ), x^2 + 1400*x + 1400 = (x + b)^2 + c ∧ c / b = -698 := by
  sorry

end quadratic_rewrite_l1555_155538


namespace max_monthly_profit_l1555_155531

/-- Represents the monthly profit as a function of price increase --/
def monthly_profit (x : ℕ) : ℤ :=
  -10 * x^2 + 110 * x + 2100

/-- The maximum allowed price increase --/
def max_increase : ℕ := 15

/-- Theorem stating the maximum monthly profit and optimal selling prices --/
theorem max_monthly_profit :
  (∃ x : ℕ, x > 0 ∧ x ≤ max_increase ∧ monthly_profit x = 2400) ∧
  (∀ x : ℕ, x > 0 ∧ x ≤ max_increase → monthly_profit x ≤ 2400) ∧
  (monthly_profit 5 = 2400 ∧ monthly_profit 6 = 2400) :=
sorry

end max_monthly_profit_l1555_155531


namespace inequality_proof_l1555_155532

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (1 / (a - b)) + (1 / (b - c)) ≥ 4 / (a - c) := by
  sorry

end inequality_proof_l1555_155532


namespace min_value_product_l1555_155547

theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 25/4 :=
by sorry

end min_value_product_l1555_155547


namespace least_multiple_13_greater_than_418_l1555_155590

theorem least_multiple_13_greater_than_418 :
  ∀ n : ℕ, n > 0 ∧ 13 ∣ n ∧ n > 418 → n ≥ 429 :=
by sorry

end least_multiple_13_greater_than_418_l1555_155590


namespace geometric_sequence_a5_l1555_155587

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_pos : ∀ n, a n > 0) 
    (h_prod : a 3 * a 7 = 64) : 
  a 5 = 8 := by
  sorry

end geometric_sequence_a5_l1555_155587


namespace function_properties_l1555_155593

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem function_properties (f g : ℝ → ℝ) (h : ∀ x, f x + g x = (1/2)^x)
  (hf : IsOdd f) (hg : IsEven g) :
  (∀ x, f x = (1/2) * (2^(-x) - 2^x)) ∧
  (∀ x, g x = (1/2) * (2^(-x) + 2^x)) ∧
  (∃ x₀ ∈ Set.Icc (1/2) 1, ∃ a : ℝ, a * f x₀ + g (2*x₀) = 0 →
    a ∈ Set.Icc (2 * Real.sqrt 2) (17/6)) := by
  sorry


end function_properties_l1555_155593


namespace hyperbola_parabola_focus_coincide_l1555_155558

theorem hyperbola_parabola_focus_coincide (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) → 
  (∃ (x y : ℝ), y^2 = 8*x) → 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1 ∧ y^2 = 8*x ∧ x = 2 ∧ y = 0) →
  a = 1 := by
sorry

end hyperbola_parabola_focus_coincide_l1555_155558


namespace quadratic_always_real_solution_l1555_155594

theorem quadratic_always_real_solution (m : ℝ) : 
  ∃ x : ℝ, x^2 - m*x + (m - 1) = 0 :=
by
  sorry

#check quadratic_always_real_solution

end quadratic_always_real_solution_l1555_155594


namespace cube_root_eight_times_sixth_root_sixtyfour_equals_four_l1555_155528

theorem cube_root_eight_times_sixth_root_sixtyfour_equals_four :
  (8 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) = 4 := by
  sorry

end cube_root_eight_times_sixth_root_sixtyfour_equals_four_l1555_155528


namespace jimmy_yellow_marbles_l1555_155535

theorem jimmy_yellow_marbles :
  ∀ (lorin_black jimmy_yellow alex_total : ℕ),
    lorin_black = 4 →
    alex_total = 19 →
    alex_total = 2 * lorin_black + (jimmy_yellow / 2) →
    jimmy_yellow = 22 :=
by
  sorry

end jimmy_yellow_marbles_l1555_155535


namespace initial_violet_balloons_count_l1555_155592

/-- The number of violet balloons Jason initially had -/
def initial_violet_balloons : ℕ := sorry

/-- The number of violet balloons Jason lost -/
def lost_violet_balloons : ℕ := 3

/-- The number of violet balloons Jason has now -/
def current_violet_balloons : ℕ := 4

/-- Theorem stating that the initial number of violet balloons is 7 -/
theorem initial_violet_balloons_count : initial_violet_balloons = 7 :=
by
  sorry

/-- Lemma showing the relationship between initial, lost, and current balloons -/
lemma balloon_relationship : initial_violet_balloons = current_violet_balloons + lost_violet_balloons :=
by
  sorry

end initial_violet_balloons_count_l1555_155592


namespace shower_water_usage_l1555_155503

theorem shower_water_usage (total : ℕ) (remy : ℕ) (h1 : total = 33) (h2 : remy = 25) :
  ∃ (M : ℕ), remy = M * (total - remy) + 1 ∧ M = 3 := by
sorry

end shower_water_usage_l1555_155503


namespace geometric_to_arithmetic_progression_l1555_155549

-- Define the four numbers
def a : ℝ := 2
def b : ℝ := 6
def c : ℝ := 18
def d : ℝ := 54

-- Theorem statement
theorem geometric_to_arithmetic_progression :
  -- The numbers form a geometric progression
  (b / a = c / b) ∧ (c / b = d / c) ∧
  -- When transformed, they form an arithmetic progression
  ((b + 4) - a = c - (b + 4)) ∧ (c - (b + 4) = (d - 28) - c) :=
by sorry

end geometric_to_arithmetic_progression_l1555_155549


namespace new_person_weight_l1555_155591

/-- Proves that if replacing a 50 kg person with a new person in a group of 5 
    increases the average weight by 4 kg, then the new person weighs 70 kg. -/
theorem new_person_weight (W : ℝ) : 
  W - 50 + (W + 20) / 5 = W + 4 → (W + 20) / 5 = 70 := by
  sorry

end new_person_weight_l1555_155591


namespace product_purchase_percentage_l1555_155540

theorem product_purchase_percentage
  (price_increase : ℝ)
  (expenditure_difference : ℝ)
  (h1 : price_increase = 0.25)
  (h2 : expenditure_difference = 0.125) :
  (1 + price_increase) * ((1 + expenditure_difference) / (1 + price_increase)) = 0.9 :=
sorry

end product_purchase_percentage_l1555_155540


namespace choose_4_from_10_l1555_155599

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by sorry

end choose_4_from_10_l1555_155599


namespace block_length_l1555_155585

/-- Calculates the length of each block given walking time, speed, and number of blocks covered -/
theorem block_length (walking_time : ℝ) (speed : ℝ) (blocks_covered : ℝ) :
  walking_time = 40 →
  speed = 100 →
  blocks_covered = 100 →
  (walking_time * speed) / blocks_covered = 40 := by
  sorry


end block_length_l1555_155585


namespace trapezium_area_l1555_155563

theorem trapezium_area (a b h : ℝ) (ha : a = 28) (hb : b = 18) (hh : h = 15) :
  (a + b) * h / 2 = 345 := by
  sorry

end trapezium_area_l1555_155563


namespace worker_c_days_l1555_155570

/-- Represents the problem of calculating the number of days worker c worked. -/
theorem worker_c_days (days_a days_b : ℕ) (wage_c : ℕ) (total_earning : ℕ) : 
  days_a = 6 →
  days_b = 9 →
  wage_c = 105 →
  total_earning = 1554 →
  ∃ (days_c : ℕ),
    (3 : ℚ) / 5 * wage_c * days_a + 
    (4 : ℚ) / 5 * wage_c * days_b + 
    wage_c * days_c = total_earning ∧
    days_c = 4 :=
by sorry

end worker_c_days_l1555_155570


namespace intersection_dot_product_l1555_155516

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (3,0)
def line_through_3_0 (l : ℝ → ℝ) : Prop := l 3 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Define the dot product of vectors OA and OB
def dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2

-- The theorem statement
theorem intersection_dot_product (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  line_through_3_0 l → intersection_points A B l → dot_product A B = 3 :=
by sorry

end intersection_dot_product_l1555_155516


namespace f_is_linear_function_l1555_155507

/-- A linear function is of the form y = kx + b, where k and b are constants, and k ≠ 0 -/
structure LinearFunction (α : Type*) [Ring α] where
  k : α
  b : α
  k_nonzero : k ≠ 0

/-- The function y = -3x + 1 -/
def f (x : ℝ) : ℝ := -3 * x + 1

/-- Theorem: f is a linear function -/
theorem f_is_linear_function : ∃ (lf : LinearFunction ℝ), ∀ x, f x = lf.k * x + lf.b :=
  sorry

end f_is_linear_function_l1555_155507


namespace complex_equation_solution_l1555_155530

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (2 : ℂ) - 3 * i * z = (4 : ℂ) + 5 * i * z ∧ z = i / 4 :=
by
  sorry

end complex_equation_solution_l1555_155530


namespace maria_juan_mm_l1555_155555

theorem maria_juan_mm (j : ℕ) (k : ℕ) (h1 : j > 0) : 
  (k * j - 3 = 2 * (j + 3)) → k = 11 := by
  sorry

end maria_juan_mm_l1555_155555


namespace binomial_n_value_l1555_155546

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem stating that for a binomial distribution with E(X) = 4 and D(X) = 2, n = 8 -/
theorem binomial_n_value (X : BinomialDistribution) 
  (h_exp : expectation X = 4)
  (h_var : variance X = 2) : 
  X.n = 8 := by sorry

end binomial_n_value_l1555_155546


namespace arccos_cos_eq_double_x_solution_l1555_155534

theorem arccos_cos_eq_double_x_solution :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → (Real.arccos (Real.cos x) = 2 * x ↔ x = 0) :=
by sorry

end arccos_cos_eq_double_x_solution_l1555_155534


namespace line_circle_intersection_l1555_155573

/-- A line passing through (-2,0) with slope k intersects the circle x^2 + y^2 = 2x at two points
    if and only if -√2/4 < k < √2/4 -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (k * x₁ - y₁ + 2*k = 0) ∧ 
    (k * x₂ - y₂ + 2*k = 0) ∧ 
    (x₁^2 + y₁^2 = 2*x₁) ∧ 
    (x₂^2 + y₂^2 = 2*x₂)) ↔ 
  (-Real.sqrt 2 / 4 < k ∧ k < Real.sqrt 2 / 4) :=
sorry

end line_circle_intersection_l1555_155573


namespace dividend_calculation_l1555_155519

theorem dividend_calculation (divisor : ℕ) (partial_quotient : ℕ) 
  (h1 : divisor = 12) 
  (h2 : partial_quotient = 909809) : 
  divisor * partial_quotient = 10917708 := by
sorry

end dividend_calculation_l1555_155519


namespace sam_has_most_pages_l1555_155586

/-- Represents a book collection --/
structure Collection where
  pagesPerInch : ℕ
  height : ℕ

/-- Calculates the total number of pages in a collection --/
def totalPages (c : Collection) : ℕ := c.pagesPerInch * c.height

theorem sam_has_most_pages (miles daphne sam : Collection)
  (h_miles : miles = ⟨5, 240⟩)
  (h_daphne : daphne = ⟨50, 25⟩)
  (h_sam : sam = ⟨30, 60⟩) :
  totalPages sam = 1800 ∧ 
  totalPages sam > totalPages miles ∧ 
  totalPages sam > totalPages daphne :=
by sorry

end sam_has_most_pages_l1555_155586


namespace sophomore_count_l1555_155515

theorem sophomore_count (total_students : ℕ) 
  (junior_percent : ℚ) (senior_percent : ℚ) (sophomore_percent : ℚ) :
  total_students = 45 →
  junior_percent = 1/5 →
  senior_percent = 3/20 →
  sophomore_percent = 1/10 →
  ∃ (juniors seniors sophomores : ℕ),
    juniors + seniors + sophomores = total_students ∧
    (junior_percent : ℚ) * juniors = (senior_percent : ℚ) * seniors ∧
    (senior_percent : ℚ) * seniors = (sophomore_percent : ℚ) * sophomores ∧
    sophomores = 21 :=
by sorry

end sophomore_count_l1555_155515


namespace solve_equation_l1555_155574

theorem solve_equation (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^5 = s^4) 
  (h3 : r - p = 31) : 
  (s : ℤ) - (q : ℤ) = -2351 := by
  sorry

end solve_equation_l1555_155574


namespace complex_number_quadrant_l1555_155501

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 - I) / (2 + 3*I) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l1555_155501


namespace jack_pounds_l1555_155509

/-- Proves that Jack has 42 pounds given the problem conditions -/
theorem jack_pounds : 
  ∀ (p : ℝ) (e : ℝ) (y : ℝ),
  e = 11 →
  y = 3000 →
  2 * e + p + y / 100 = 9400 / 100 →
  p = 42 := by
  sorry


end jack_pounds_l1555_155509


namespace area_geometric_mean_l1555_155500

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define a point on a line
def pointOnLine (p1 p2 : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := sorry

-- Define a right-angled triangle
def isRightAngled (t : Triangle) : Prop := sorry

theorem area_geometric_mean 
  (ABC : Triangle) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (h1 : area ABC = S₁) 
  (O : ℝ × ℝ) 
  (h2 : O = orthocenter ABC) 
  (AOB : Triangle) 
  (h3 : area AOB = S₂) 
  (K : ℝ × ℝ) 
  (h4 : ∃ k, K = pointOnLine O ABC.C k) 
  (ABK : Triangle) 
  (h5 : isRightAngled ABK) : 
  area ABK = Real.sqrt (S₁ * S₂) := 
by sorry

end area_geometric_mean_l1555_155500


namespace lines_in_plane_not_intersecting_are_parallel_l1555_155569

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

theorem lines_in_plane_not_intersecting_are_parallel 
  (α : Plane3D) (a b : Line3D) 
  (ha : contained_in a α) 
  (hb : contained_in b α) 
  (hnot_intersect : ¬ intersect a b) : 
  parallel a b :=
sorry

end lines_in_plane_not_intersecting_are_parallel_l1555_155569


namespace AB_length_l1555_155552

-- Define the points and lengths
variable (A B C D E F G : ℝ)

-- Define the midpoint relationships
axiom C_midpoint : C = (A + B) / 2
axiom D_midpoint : D = (A + C) / 2
axiom E_midpoint : E = (A + D) / 2
axiom F_midpoint : F = (A + E) / 2
axiom G_midpoint : G = (A + F) / 2

-- Given condition
axiom AG_length : G - A = 5

-- Theorem to prove
theorem AB_length : B - A = 160 := by
  sorry

end AB_length_l1555_155552


namespace fraction_sum_proof_l1555_155582

theorem fraction_sum_proof : 
  (1 / 12 : ℚ) + (2 / 12 : ℚ) + (3 / 12 : ℚ) + (4 / 12 : ℚ) + (5 / 12 : ℚ) + 
  (6 / 12 : ℚ) + (7 / 12 : ℚ) + (8 / 12 : ℚ) + (9 / 12 : ℚ) + (65 / 12 : ℚ) + 
  (3 / 4 : ℚ) = 119 / 12 := by
  sorry

end fraction_sum_proof_l1555_155582


namespace dawn_savings_percentage_l1555_155522

theorem dawn_savings_percentage (annual_salary : ℕ) (monthly_savings : ℕ) : annual_salary = 48000 → monthly_savings = 400 → (monthly_savings : ℚ) / ((annual_salary : ℚ) / 12) = 1/10 := by
  sorry

end dawn_savings_percentage_l1555_155522


namespace mixed_doubles_selections_l1555_155578

/-- The number of male players in the table tennis team -/
def num_male_players : ℕ := 5

/-- The number of female players in the table tennis team -/
def num_female_players : ℕ := 4

/-- The total number of ways to select a mixed doubles team -/
def total_selections : ℕ := num_male_players * num_female_players

theorem mixed_doubles_selections :
  total_selections = 20 :=
sorry

end mixed_doubles_selections_l1555_155578


namespace max_distance_complex_l1555_155521

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (max_dist : ℝ), max_dist = 8 * (Real.sqrt 29 + 2) ∧
  ∀ (w : ℂ), Complex.abs w = 2 →
    Complex.abs ((5 + 2*I)*w^3 - w^4) ≤ max_dist :=
by sorry

end max_distance_complex_l1555_155521


namespace basketball_lineup_combinations_l1555_155577

theorem basketball_lineup_combinations : 
  ∀ (total_players : ℕ) (fixed_players : ℕ) (lineup_size : ℕ),
    total_players = 15 →
    fixed_players = 2 →
    lineup_size = 6 →
    Nat.choose (total_players - fixed_players) (lineup_size - fixed_players) = 715 := by
  sorry

end basketball_lineup_combinations_l1555_155577


namespace unique_rectangle_from_rods_l1555_155571

theorem unique_rectangle_from_rods (n : ℕ) (h : n = 22) : 
  (∃! (l w : ℕ), l + w = n / 2 ∧ l * 2 + w * 2 = n ∧ l > 0 ∧ w > 0) :=
by sorry

end unique_rectangle_from_rods_l1555_155571


namespace part_one_part_two_l1555_155584

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + a*x₁ + 1/16 = 0 ∧ x₂^2 + a*x₂ + 1/16 = 0

def q (a : ℝ) : Prop := 1/a > 1

-- Theorem for part (1)
theorem part_one (a : ℝ) : p a → a > 1/2 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≥ 1 ∨ (0 < a ∧ a ≤ 1/2) := by sorry

end part_one_part_two_l1555_155584


namespace trigonometric_problem_l1555_155512

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_distance : sqrt ((cos α - cos β)^2 + (sin α - sin β)^2) = sqrt 10 / 5)
  (h_tan : tan (α/2) = 1/2) :
  cos (α - β) = 4/5 ∧ cos α = 3/5 ∧ cos β = 24/25 := by
  sorry

end trigonometric_problem_l1555_155512


namespace expression_evaluation_l1555_155551

theorem expression_evaluation : -1^2008 + (-1)^2009 + 1^2010 + (-1)^2011 + 1^2012 = -1 := by
  sorry

end expression_evaluation_l1555_155551


namespace corn_stalk_calculation_hilary_corn_stalks_l1555_155524

theorem corn_stalk_calculation (ears_per_stalk : ℕ) 
  (kernels_low : ℕ) (kernels_high : ℕ) (total_kernels : ℕ) : ℕ :=
  let avg_kernels := (kernels_low + kernels_high) / 2
  let total_ears := total_kernels / avg_kernels
  total_ears / ears_per_stalk

theorem hilary_corn_stalks : 
  corn_stalk_calculation 4 500 600 237600 = 108 := by
  sorry

end corn_stalk_calculation_hilary_corn_stalks_l1555_155524


namespace smallest_y_for_perfect_cube_l1555_155543

def x : ℕ := 5 * 16 * 27

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by sorry

end smallest_y_for_perfect_cube_l1555_155543


namespace arithmetic_geometric_sequence_properties_l1555_155505

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def geometric_subsequence (a : ℕ → ℝ) (k : ℕ → ℕ) (q : ℝ) : Prop :=
  ∀ n, a (k (n + 1)) = a (k n) * q

def strictly_increasing (k : ℕ → ℕ) : Prop :=
  ∀ n, k n < k (n + 1)

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ) (d q : ℝ) (k : ℕ → ℕ) 
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_subsequence a k q)
  (h_incr : strictly_increasing k)
  (h_d_neq_0 : d ≠ 0)
  (h_k1 : k 1 = 1)
  (h_k2 : k 2 = 3)
  (h_k3 : k 3 = 8) :
  (a 1 / d = 4 / 3) ∧ 
  ((∀ n, k (n + 1) = k n * q) ↔ a 1 / d = 1) ∧
  ((∀ n, k (n + 1) = k n * q) → 
   (∀ n : ℕ, 0 < n → a n + a (k n) > 2 * k n) → 
   a 1 ≥ 2) :=
sorry

end arithmetic_geometric_sequence_properties_l1555_155505


namespace triangle_sine_b_l1555_155561

theorem triangle_sine_b (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle angle condition
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  a = 2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) → -- Law of sines
  b = 2 * Real.sin (A/2) * Real.sin (C/2) → -- Law of sines
  c = 2 * Real.sin (A/2) * Real.sin (B/2) → -- Law of sines
  a + c = 2*b → -- Given condition
  A - C = π/3 → -- Given condition
  Real.sin B = Real.sqrt 39 / 8 := by sorry

end triangle_sine_b_l1555_155561


namespace digit_2023_of_17_19_l1555_155510

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit (n d k : ℕ) : ℕ := sorry

theorem digit_2023_of_17_19 : nth_digit 17 19 2023 = 3 := by sorry

end digit_2023_of_17_19_l1555_155510


namespace passengers_after_first_stop_l1555_155506

/-- 
Given a train with an initial number of passengers and some passengers getting off at the first stop,
this theorem proves the number of passengers remaining after the first stop.
-/
theorem passengers_after_first_stop 
  (initial_passengers : ℕ) 
  (passengers_left : ℕ) 
  (h1 : initial_passengers = 48)
  (h2 : passengers_left = initial_passengers - 17) : 
  passengers_left = 31 := by
  sorry

end passengers_after_first_stop_l1555_155506


namespace product_of_roots_cubic_equation_l1555_155537

theorem product_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 2 * x^3 - 7 * x^2 - 6
  let roots := {r : ℝ | f r = 0}
  ∀ r s t : ℝ, r ∈ roots → s ∈ roots → t ∈ roots → r * s * t = 3 :=
by
  sorry

end product_of_roots_cubic_equation_l1555_155537


namespace football_cost_proof_l1555_155553

def shorts_cost : ℝ := 2.40
def shoes_cost : ℝ := 11.85
def zachary_has : ℝ := 10
def zachary_needs : ℝ := 8

def total_cost : ℝ := zachary_has + zachary_needs

def football_cost : ℝ := total_cost - shorts_cost - shoes_cost

theorem football_cost_proof : football_cost = 3.75 := by
  sorry

end football_cost_proof_l1555_155553


namespace unique_a_value_l1555_155541

/-- The function f(x) = ax³ - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x + 1

/-- The theorem stating that a = 4 is the unique value satisfying the condition -/
theorem unique_a_value : ∃! a : ℝ, ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≥ 0 :=
by
  -- The proof goes here
  sorry

end unique_a_value_l1555_155541


namespace square_garden_multiple_l1555_155523

theorem square_garden_multiple (a p : ℝ) (h1 : p = 38) (h2 : a = (p / 4)^2) (h3 : ∃ m : ℝ, a = m * p + 14.25) : 
  ∃ m : ℝ, a = m * p + 14.25 ∧ m = 2 :=
sorry

end square_garden_multiple_l1555_155523


namespace josh_selena_distance_ratio_l1555_155554

/-- Proves that the ratio of Josh's distance to Selena's distance is 1/2 -/
theorem josh_selena_distance_ratio :
  let total_distance : ℝ := 36
  let selena_distance : ℝ := 24
  let josh_distance : ℝ := total_distance - selena_distance
  josh_distance / selena_distance = 1 / 2 := by
  sorry

end josh_selena_distance_ratio_l1555_155554


namespace cost_price_calculation_l1555_155542

theorem cost_price_calculation (selling_price_profit selling_price_loss : ℕ) 
  (h : selling_price_profit - selling_price_loss = 2 * (selling_price_profit - 50)) :
  50 = (selling_price_profit + selling_price_loss) / 2 := by
  sorry

#check cost_price_calculation 57 43

end cost_price_calculation_l1555_155542


namespace rhombus_diagonal_sum_squares_l1555_155583

/-- A rhombus with side length 2 has the sum of squares of its diagonals equal to 16 -/
theorem rhombus_diagonal_sum_squares (d₁ d₂ : ℝ) : 
  d₁ > 0 → d₂ > 0 → (d₁ / 2) ^ 2 + (d₂ / 2) ^ 2 = 2 ^ 2 → d₁ ^ 2 + d₂ ^ 2 = 16 := by
  sorry

end rhombus_diagonal_sum_squares_l1555_155583


namespace fraction_simplification_l1555_155520

theorem fraction_simplification :
  (6 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + 2 * Real.sqrt 18) = (3 * Real.sqrt 2) / 17 := by
  sorry

end fraction_simplification_l1555_155520


namespace sufficient_not_necessary_l1555_155581

theorem sufficient_not_necessary (x₁ x₂ : ℝ) :
  (∀ x₁ x₂ : ℝ, (x₁ > 1 ∧ x₂ > 1) → (x₁ + x₂ > 2 ∧ x₁ * x₂ > 1)) ∧
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ > 2 ∧ x₁ * x₂ > 1) ∧ ¬(x₁ > 1 ∧ x₂ > 1)) :=
by sorry

end sufficient_not_necessary_l1555_155581


namespace cd_purchase_cost_l1555_155566

/-- Calculates the total cost of purchasing CDs -/
def total_cost (life_journey_price : ℕ) (day_life_price : ℕ) (rescind_price : ℕ) (quantity : ℕ) : ℕ :=
  quantity * (life_journey_price + day_life_price + rescind_price)

/-- Theorem: The total cost of buying 3 CDs each of The Life Journey ($100), 
    A Day a Life ($50), and When You Rescind ($85) is $705 -/
theorem cd_purchase_cost : total_cost 100 50 85 3 = 705 := by
  sorry

end cd_purchase_cost_l1555_155566


namespace midpoint_distance_theorem_l1555_155564

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 4, -3)
  let B : ℝ × ℝ := (-6, 2*t + 5)
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (M.1 - B.1)^2 + (M.2 - B.2)^2 = 4*t^2 + 3*t →
  t = (7 + Real.sqrt 185) / 4 ∨ t = (7 - Real.sqrt 185) / 4 :=
by sorry

end midpoint_distance_theorem_l1555_155564


namespace expression_equals_53_l1555_155572

theorem expression_equals_53 : (-6)^4 / 6^2 + 2^5 - 6^1 - 3^2 = 53 := by
  sorry

end expression_equals_53_l1555_155572


namespace f_monotone_decreasing_l1555_155539

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_monotone_decreasing :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.exp (-1)),
    StrictMonoOn f (Set.Ioo 0 (Real.exp (-1))) :=
by
  sorry

end f_monotone_decreasing_l1555_155539


namespace angle_610_equivalent_l1555_155580

def same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₁ = θ₂ + k * 360

theorem angle_610_equivalent :
  ∀ k : ℤ, same_terminal_side 610 (k * 360 + 250) := by sorry

end angle_610_equivalent_l1555_155580


namespace problem_1_l1555_155533

theorem problem_1 : (-16) - 25 + (-43) - (-39) = -45 := by sorry

end problem_1_l1555_155533


namespace inequality_solution_l1555_155557

theorem inequality_solution (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) (h4 : x ≠ 4) (h5 : x ≠ 5) :
  (2 / (x - 1) - 3 / (x - 3) + 2 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < -1 ∨ (1 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (7 < x ∧ x < 8)) :=
by sorry

end inequality_solution_l1555_155557


namespace inequality_proof_l1555_155589

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 / b) + (c^2 / d) ≥ ((a + c)^2) / (b + d) := by
  sorry

end inequality_proof_l1555_155589


namespace sin_product_identity_l1555_155595

theorem sin_product_identity : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (60 * π / 180) * Real.sin (72 * π / 180) = 
  ((Real.sqrt 5 + 1) * Real.sqrt 3) / 16 := by
sorry

end sin_product_identity_l1555_155595


namespace shekar_average_marks_l1555_155576

def shekar_scores : List ℕ := [76, 65, 82, 62, 85]

theorem shekar_average_marks :
  (shekar_scores.sum : ℚ) / shekar_scores.length = 74 := by sorry

end shekar_average_marks_l1555_155576


namespace fish_in_pond_l1555_155504

/-- Approximates the total number of fish in a pond based on a tag-and-recapture method. -/
def approximate_fish_count (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (initial_tagged * second_catch) / tagged_in_second

/-- The approximate number of fish in the pond given the tag-and-recapture data. -/
theorem fish_in_pond (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ)
    (h1 : initial_tagged = 50)
    (h2 : second_catch = 50)
    (h3 : tagged_in_second = 10) :
  approximate_fish_count initial_tagged second_catch tagged_in_second = 250 := by
  sorry

#eval approximate_fish_count 50 50 10

end fish_in_pond_l1555_155504


namespace sum_of_two_sequences_l1555_155579

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_two_sequences : 
  let seq1 := arithmetic_sequence 2 12 4
  let seq2 := arithmetic_sequence 18 12 4
  sum_list (seq1 ++ seq2) = 224 := by
sorry

end sum_of_two_sequences_l1555_155579


namespace sandwich_optimization_l1555_155596

/-- Represents the number of sandwiches of each type -/
structure SandwichCount where
  cheese : ℕ
  salami : ℕ

/-- Represents the available resources -/
structure Resources where
  bread : ℕ  -- in dkg
  butter : ℕ -- in dkg
  cheese : ℕ -- in dkg
  salami : ℕ -- in dkg

/-- Represents the ingredient requirements for each sandwich type -/
structure SandwichRequirements where
  cheese_bread : ℕ  -- in dkg
  cheese_butter : ℕ -- in dkg
  cheese_cheese : ℕ -- in dkg
  salami_bread : ℕ  -- in dkg
  salami_butter : ℕ -- in dkg
  salami_salami : ℕ -- in dkg

def is_valid_sandwich_count (count : SandwichCount) (resources : Resources) 
    (requirements : SandwichRequirements) : Prop :=
  count.cheese * requirements.cheese_bread + count.salami * requirements.salami_bread ≤ resources.bread ∧
  count.cheese * requirements.cheese_butter + count.salami * requirements.salami_butter ≤ resources.butter ∧
  count.cheese * requirements.cheese_cheese ≤ resources.cheese ∧
  count.salami * requirements.salami_salami ≤ resources.salami

def total_sandwiches (count : SandwichCount) : ℕ :=
  count.cheese + count.salami

def revenue (count : SandwichCount) (cheese_price salami_price : ℚ) : ℚ :=
  count.cheese * cheese_price + count.salami * salami_price

def preparation_time (count : SandwichCount) (cheese_time salami_time : ℕ) : ℕ :=
  count.cheese * cheese_time + count.salami * salami_time

theorem sandwich_optimization (resources : Resources) 
    (requirements : SandwichRequirements) 
    (cheese_price salami_price : ℚ) 
    (cheese_time salami_time : ℕ) :
    ∃ (max_count optimal_revenue_count optimal_time_count : SandwichCount),
      is_valid_sandwich_count max_count resources requirements ∧
      total_sandwiches max_count = 40 ∧
      (∀ count, is_valid_sandwich_count count resources requirements → 
        total_sandwiches count ≤ total_sandwiches max_count) ∧
      is_valid_sandwich_count optimal_revenue_count resources requirements ∧
      revenue optimal_revenue_count cheese_price salami_price = 63.5 ∧
      (∀ count, is_valid_sandwich_count count resources requirements → 
        revenue count cheese_price salami_price ≤ revenue optimal_revenue_count cheese_price salami_price) ∧
      is_valid_sandwich_count optimal_time_count resources requirements ∧
      total_sandwiches optimal_time_count = 40 ∧
      preparation_time optimal_time_count cheese_time salami_time = 50 ∧
      (∀ count, is_valid_sandwich_count count resources requirements ∧ total_sandwiches count = 40 → 
        preparation_time optimal_time_count cheese_time salami_time ≤ preparation_time count cheese_time salami_time) :=
  sorry

end sandwich_optimization_l1555_155596


namespace sufficient_not_necessary_l1555_155518

theorem sufficient_not_necessary : 
  (∃ m : ℝ, m = 9 → m > 8) ∧ 
  (∃ m : ℝ, m > 8 ∧ m ≠ 9) := by
  sorry

end sufficient_not_necessary_l1555_155518


namespace race_time_difference_l1555_155575

/-- Proves that the difference in time taken by two teams to complete a 300-mile course is 3 hours,
    given that one team's speed is 5 mph greater than the other team's speed of 20 mph. -/
theorem race_time_difference (distance : ℝ) (speed_E : ℝ) (speed_diff : ℝ) : 
  distance = 300 → 
  speed_E = 20 → 
  speed_diff = 5 → 
  distance / speed_E - distance / (speed_E + speed_diff) = 3 := by
sorry

end race_time_difference_l1555_155575


namespace prize_distribution_correct_l1555_155527

/-- Represents the prize distribution and cost calculation for a school event. -/
def prize_distribution (x : ℕ) : Prop :=
  let first_prize := x
  let second_prize := 4 * x - 10
  let third_prize := 90 - 5 * x
  let total_prizes := first_prize + second_prize + third_prize
  let total_cost := 18 * first_prize + 12 * second_prize + 6 * third_prize
  (total_prizes = 80) ∧ 
  (total_cost = 420 + 36 * x) ∧
  (x = 12 → total_cost = 852)

/-- Theorem stating the correctness of the prize distribution and cost calculation. -/
theorem prize_distribution_correct : 
  ∀ x : ℕ, prize_distribution x := by sorry

end prize_distribution_correct_l1555_155527


namespace second_candidate_votes_l1555_155567

theorem second_candidate_votes
  (total_votes : ℕ)
  (first_candidate_percentage : ℚ)
  (h1 : total_votes = 1200)
  (h2 : first_candidate_percentage = 80 / 100) :
  (1 - first_candidate_percentage) * total_votes = 240 :=
by sorry

end second_candidate_votes_l1555_155567


namespace div_point_five_by_point_zero_twenty_five_l1555_155598

theorem div_point_five_by_point_zero_twenty_five : (0.5 : ℚ) / 0.025 = 20 := by
  sorry

end div_point_five_by_point_zero_twenty_five_l1555_155598


namespace parallelogram_distance_l1555_155550

/-- Given a parallelogram with the following properties:
    - One side has length 20 feet
    - The perpendicular distance between that side and its opposite side is 60 feet
    - The other two parallel sides are each 50 feet long
    Prove that the perpendicular distance between the 50-foot sides is 24 feet. -/
theorem parallelogram_distance (base : ℝ) (height : ℝ) (side : ℝ) (h1 : base = 20) 
    (h2 : height = 60) (h3 : side = 50) : 
  (base * height) / side = 24 := by
sorry

end parallelogram_distance_l1555_155550


namespace existence_of_special_numbers_l1555_155513

/-- Check if a number uses only the digits 1, 2, 3, 4, 5 --/
def usesValidDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5]

/-- Check if two numbers use all the digits 1, 2, 3, 4, 5 exactly once between them --/
def useAllDigitsOnce (a b : ℕ) : Prop :=
  (a.digits 10 ++ b.digits 10).toFinset = {1, 2, 3, 4, 5}

theorem existence_of_special_numbers : ∃ a b : ℕ,
  10 ≤ a ∧ a < 100 ∧
  100 ≤ b ∧ b < 1000 ∧
  usesValidDigits a ∧
  usesValidDigits b ∧
  useAllDigitsOnce a b ∧
  b % a = 0 :=
by sorry

end existence_of_special_numbers_l1555_155513


namespace sum_of_special_numbers_l1555_155568

-- Define the smallest odd prime number
def smallest_odd_prime : ℕ := 3

-- Define the largest integer less than 150 with exactly three positive divisors
def largest_three_divisor_under_150 : ℕ := 121

-- Theorem statement
theorem sum_of_special_numbers : 
  smallest_odd_prime + largest_three_divisor_under_150 = 124 := by sorry

end sum_of_special_numbers_l1555_155568


namespace hexagon_fills_ground_l1555_155525

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

def can_fill_ground (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * interior_angle n = 360

theorem hexagon_fills_ground :
  can_fill_ground 6 ∧
  ¬ can_fill_ground 10 ∧
  ¬ can_fill_ground 8 ∧
  ¬ can_fill_ground 5 := by sorry

end hexagon_fills_ground_l1555_155525


namespace banana_arrangements_count_l1555_155517

def banana_arrangements : ℕ :=
  Nat.factorial 6 / Nat.factorial 3

theorem banana_arrangements_count : banana_arrangements = 120 := by
  sorry

end banana_arrangements_count_l1555_155517


namespace average_increase_fraction_l1555_155544

-- Define the number of students in the class
def num_students : ℕ := 80

-- Define the correct mark and the wrongly entered mark
def correct_mark : ℕ := 62
def wrong_mark : ℕ := 82

-- Define the increase in total marks due to the error
def mark_difference : ℕ := wrong_mark - correct_mark

-- State the theorem
theorem average_increase_fraction :
  (mark_difference : ℚ) / num_students = 1 / 4 := by
  sorry

end average_increase_fraction_l1555_155544


namespace smallest_perfect_square_div_by_5_and_6_l1555_155514

theorem smallest_perfect_square_div_by_5_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  (∀ k : ℕ, k > 0 → (∃ m : ℕ, k = m^2) → k % 5 = 0 → k % 6 = 0 → k ≥ n) ∧
  n = 900 := by
sorry

end smallest_perfect_square_div_by_5_and_6_l1555_155514


namespace tank_capacity_proof_l1555_155556

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 675

/-- The time in minutes for pipe A to fill the tank -/
def pipe_a_time : ℝ := 12

/-- The time in minutes for pipe B to fill the tank -/
def pipe_b_time : ℝ := 20

/-- The rate at which pipe C drains water in liters per minute -/
def pipe_c_rate : ℝ := 45

/-- The time in minutes to fill the tank when all pipes are opened -/
def all_pipes_time : ℝ := 15

/-- Theorem stating that the tank capacity is correct given the conditions -/
theorem tank_capacity_proof :
  tank_capacity = pipe_a_time * pipe_b_time * all_pipes_time * pipe_c_rate /
    (pipe_a_time * pipe_b_time - pipe_a_time * all_pipes_time - pipe_b_time * all_pipes_time) :=
by sorry

end tank_capacity_proof_l1555_155556


namespace shaded_region_correct_l1555_155548

-- Define the universal set U and subsets A and B
variable (U : Type) (A B : Set U)

-- Define the shaded region
def shaded_region (U : Type) (A B : Set U) : Set U :=
  (Aᶜ) ∩ (Bᶜ)

-- Theorem statement
theorem shaded_region_correct (U : Type) (A B : Set U) :
  shaded_region U A B = (Aᶜ) ∩ (Bᶜ) :=
by
  -- The proof would go here
  sorry

end shaded_region_correct_l1555_155548


namespace tangent_line_exists_l1555_155545

theorem tangent_line_exists (k : ℝ) : 
  ∃ q : ℝ, ∃ x y : ℝ, 
    (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1 ∧ 
    y = k * x ∧
    ∀ x' y' : ℝ, (x' + Real.cos q)^2 + (y' - Real.sin q)^2 = 1 → 
      y' = k * x' → (x' = x ∧ y' = y) :=
by sorry

end tangent_line_exists_l1555_155545


namespace max_product_constraint_l1555_155502

theorem max_product_constraint (m n : ℝ) (hm : m > 0) (hn : n > 0) (hsum : m + n = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 4 → x * y ≤ m * n → m * n = 4 := by
sorry

end max_product_constraint_l1555_155502


namespace eulerian_path_implies_at_most_two_odd_vertices_l1555_155529

/-- A simple graph. -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A vertex has odd degree if its degree is odd. -/
def hasOddDegree (G : Graph V) (v : V) : Prop :=
  Odd (degree G v)

/-- An Eulerian path in a graph. -/
def hasEulerianPath (G : Graph V) : Prop := sorry

/-- The main theorem: If a graph has an Eulerian path, 
    then the number of vertices with odd degree is at most 2. -/
theorem eulerian_path_implies_at_most_two_odd_vertices 
  (V : Type*) (G : Graph V) : 
  hasEulerianPath G → 
  ∃ (n : ℕ), n ≤ 2 ∧ (∃ (S : Finset V), S.card = n ∧ 
    ∀ v, v ∈ S ↔ hasOddDegree G v) := by
  sorry

end eulerian_path_implies_at_most_two_odd_vertices_l1555_155529


namespace probability_between_C_and_D_l1555_155508

/-- Given points A, B, C, D on a line segment AB where AB = 4AD and AB = 5BC,
    prove that the probability of a randomly selected point on AB
    being between C and D is 11/20. -/
theorem probability_between_C_and_D (A B C D : ℝ) : 
  A < C ∧ C < D ∧ D < B →  -- Points are in order on the line
  (B - A) = 4 * (D - A) →  -- AB = 4AD
  (B - A) = 5 * (C - B) →  -- AB = 5BC
  (D - C) / (B - A) = 11 / 20 := by
  sorry

end probability_between_C_and_D_l1555_155508


namespace right_triangle_area_l1555_155511

/-- The area of a right triangle with base 30 and height 24 is 360 -/
theorem right_triangle_area :
  let base : ℝ := 30
  let height : ℝ := 24
  (1 / 2 : ℝ) * base * height = 360 :=
by sorry

end right_triangle_area_l1555_155511


namespace geometric_sequence_common_ratio_l1555_155536

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h_sum : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h_eq1 : 2 * (a 6) = 3 * (S 4) + 1) 
  (h_eq2 : a 7 = 3 * (S 5) + 1) : 
  q = 3 := by
  sorry

end geometric_sequence_common_ratio_l1555_155536


namespace system_solution_l1555_155526

theorem system_solution : ∃ (x y : ℚ), 
  (x * (1/7)^2 = 7^3) ∧ 
  (x + y = 7^2) ∧ 
  (x = 16807) ∧ 
  (y = -16758) := by
sorry

end system_solution_l1555_155526


namespace sons_age_l1555_155597

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end sons_age_l1555_155597


namespace total_items_proof_l1555_155588

def days : ℕ := 10

def pebble_sequence (n : ℕ) : ℕ := n

def seashell_sequence (n : ℕ) : ℕ := 2 * n - 1

def total_items : ℕ := (days * (pebble_sequence 1 + pebble_sequence days)) / 2 +
                       (days * (seashell_sequence 1 + seashell_sequence days)) / 2

theorem total_items_proof : total_items = 155 := by
  sorry

end total_items_proof_l1555_155588


namespace snake_sum_squares_geq_n_squared_l1555_155559

/-- Represents a snake (python or anaconda) in the grid -/
structure Snake where
  length : ℕ
  is_python : Bool

/-- Represents the n×n grid with snakes -/
structure Grid (n : ℕ) where
  snakes : List Snake
  valid : Bool

/-- The sum of squares of snake lengths -/
def sum_of_squares (grid : Grid n) : ℕ :=
  grid.snakes.map (λ s => s.length * s.length) |>.sum

/-- The theorem to be proved -/
theorem snake_sum_squares_geq_n_squared (n : ℕ) (grid : Grid n) 
  (h1 : n > 0)
  (h2 : grid.valid)
  (h3 : grid.snakes.length > 0) :
  sum_of_squares grid ≥ n * n := by
  sorry


end snake_sum_squares_geq_n_squared_l1555_155559
