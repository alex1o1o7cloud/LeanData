import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_lower_bound_l1895_189528

theorem sum_of_roots_lower_bound (k : ℝ) (α β : ℝ) : 
  (∃ x : ℝ, x^2 - 2*(1-k)*x + k^2 = 0) →
  (α^2 - 2*(1-k)*α + k^2 = 0) →
  (β^2 - 2*(1-k)*β + k^2 = 0) →
  α + β ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_lower_bound_l1895_189528


namespace NUMINAMATH_CALUDE_window_purchase_savings_l1895_189575

/-- Calculates the cost of purchasing windows with a discount after the first five -/
def calculateCost (regularPrice : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  if quantity ≤ 5 then
    regularPrice * quantity
  else
    regularPrice * 5 + (regularPrice - discount) * (quantity - 5)

theorem window_purchase_savings :
  let regularPrice : ℕ := 120
  let discount : ℕ := 20
  let daveWindows : ℕ := 10
  let dougWindows : ℕ := 13
  let daveCost := calculateCost regularPrice discount daveWindows
  let dougCost := calculateCost regularPrice discount dougWindows
  let jointCost := calculateCost regularPrice discount (daveWindows + dougWindows)
  daveCost + dougCost - jointCost = 100 := by sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l1895_189575


namespace NUMINAMATH_CALUDE_factorization_proof_l1895_189560

theorem factorization_proof (y : ℝ) : 4*y*(y+2) + 9*(y+2) + 2*(y+2) = (y+2)*(4*y+11) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1895_189560


namespace NUMINAMATH_CALUDE_log_inequality_l1895_189538

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1895_189538


namespace NUMINAMATH_CALUDE_x_value_l1895_189509

theorem x_value : ∃ x : ℝ, 3 * x = (26 - x) + 18 ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_x_value_l1895_189509


namespace NUMINAMATH_CALUDE_divisibility_sequence_l1895_189526

theorem divisibility_sequence (a : ℕ) : ∃ n : ℕ, ∀ k : ℕ, a ∣ (n^(n^k) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_sequence_l1895_189526


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1895_189571

/-- Sum of a geometric series with n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the geometric series -/
def a : ℚ := 1/4

/-- Common ratio of the geometric series -/
def r : ℚ := 1/4

/-- Number of terms to sum -/
def n : ℕ := 6

theorem geometric_series_sum :
  geometricSum a r n = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1895_189571


namespace NUMINAMATH_CALUDE_four_balls_three_boxes_l1895_189579

/-- The number of ways to put n different balls into k boxes -/
def ways_to_put_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: Putting 4 different balls into 3 boxes results in 81 different ways -/
theorem four_balls_three_boxes : ways_to_put_balls 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_balls_three_boxes_l1895_189579


namespace NUMINAMATH_CALUDE_triangle_inequality_third_stick_length_l1895_189597

theorem triangle_inequality (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (a < b + c ∧ b < c + a ∧ c < a + b) :=
sorry

theorem third_stick_length (a b : ℝ) (ha : a = 20) (hb : b = 30) :
  ∃ c, c = 30 ∧ 
       (a + b > c ∧ b + c > a ∧ c + a > b) ∧
       ¬(a + b > 10 ∧ b + 10 > a ∧ 10 + a > b) ∧
       ¬(a + b > 50 ∧ b + 50 > a ∧ 50 + a > b) ∧
       ¬(a + b > 70 ∧ b + 70 > a ∧ 70 + a > b) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_stick_length_l1895_189597


namespace NUMINAMATH_CALUDE_irreducible_fractions_l1895_189518

theorem irreducible_fractions (a b m n : ℕ) (h_n : n > 0) :
  (Nat.gcd a b = 1 → Nat.gcd (b - a) b = 1) ∧
  (Nat.gcd (m - n) (m + n) = 1 → Nat.gcd m n = 1) ∧
  (∃ (k : ℕ), (5 * n + 2) = k * (10 * n + 7) → Nat.gcd (5 * n + 2) (10 * n + 7) = 3) :=
by sorry

end NUMINAMATH_CALUDE_irreducible_fractions_l1895_189518


namespace NUMINAMATH_CALUDE_parallelogram_side_lengths_l1895_189522

def parallelogram_properties (angle : ℝ) (shorter_diagonal : ℝ) (perpendicular : ℝ) : Prop :=
  angle = 60 ∧ 
  shorter_diagonal = 2 * Real.sqrt 31 ∧ 
  perpendicular = Real.sqrt 75 / 2

theorem parallelogram_side_lengths 
  (angle : ℝ) (shorter_diagonal : ℝ) (perpendicular : ℝ) 
  (h : parallelogram_properties angle shorter_diagonal perpendicular) :
  ∃ (longer_side shorter_side longer_diagonal : ℝ),
    longer_side = 12 ∧ 
    shorter_side = 10 ∧ 
    longer_diagonal = 2 * Real.sqrt 91 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_side_lengths_l1895_189522


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l1895_189511

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_sufficient_not_necessary
  (l m : Line) (α : Plane)
  (h_different : l ≠ m)
  (h_parallel : parallel m α) :
  (∀ l m α, perpendicular_to_plane l α → perpendicular l m) ∧
  (∃ l m α, perpendicular l m ∧ ¬ perpendicular_to_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l1895_189511


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1895_189583

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∀ x y : ℝ, (2*a + 5)*x + (a - 2)*y + 4 = 0 ∧ (2 - a)*x + (a + 3)*y - 1 = 0 → 
    ((2*a + 5)*(2 - a) + (a - 2)*(a + 3) = 0)) → 
  (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1895_189583


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l1895_189510

/-- The third smallest positive integer divisible by all integers less than 9 -/
def M : ℕ := sorry

/-- M is divisible by all positive integers less than 9 -/
axiom M_divisible (n : ℕ) (h : n > 0 ∧ n < 9) : M % n = 0

/-- M is the third smallest such integer -/
axiom M_third_smallest :
  ∀ k : ℕ, k > 0 ∧ k < M → (∀ n : ℕ, n > 0 ∧ n < 9 → k % n = 0) →
  ∃ j : ℕ, j > 0 ∧ j < M ∧ j ≠ k ∧ (∀ n : ℕ, n > 0 ∧ n < 9 → j % n = 0)

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem sum_of_digits_M : sum_of_digits M = 9 := sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l1895_189510


namespace NUMINAMATH_CALUDE_kim_average_unchanged_l1895_189505

def kim_scores : List ℝ := [92, 86, 95, 89, 93]

theorem kim_average_unchanged (scores := kim_scores) :
  let first_three_avg := (scores.take 3).sum / 3
  let all_five_avg := scores.sum / 5
  all_five_avg - first_three_avg = 0 := by
sorry

end NUMINAMATH_CALUDE_kim_average_unchanged_l1895_189505


namespace NUMINAMATH_CALUDE_total_population_avalon_l1895_189567

theorem total_population_avalon (num_towns : ℕ) (avg_lower avg_upper : ℝ) :
  num_towns = 25 →
  5400 ≤ avg_lower →
  avg_upper ≤ 5700 →
  avg_lower ≤ (avg_lower + avg_upper) / 2 →
  (avg_lower + avg_upper) / 2 ≤ avg_upper →
  ∃ (total_population : ℝ),
    total_population = num_towns * ((avg_lower + avg_upper) / 2) ∧
    total_population = 138750 :=
by sorry

end NUMINAMATH_CALUDE_total_population_avalon_l1895_189567


namespace NUMINAMATH_CALUDE_biased_coin_theorem_l1895_189589

def biased_coin_prob (h : ℚ) : Prop :=
  (15 : ℚ) * h^2 * (1 - h)^4 = (20 : ℚ) * h^3 * (1 - h)^3

theorem biased_coin_theorem :
  ∀ h : ℚ, 0 < h → h < 1 → biased_coin_prob h →
  (15 : ℚ) * h^4 * (1 - h)^2 = 40 / 243 :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_theorem_l1895_189589


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l1895_189563

theorem added_number_after_doubling (initial_number : ℕ) (added_number : ℕ) : 
  initial_number = 9 →
  3 * (2 * initial_number + added_number) = 93 →
  added_number = 13 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l1895_189563


namespace NUMINAMATH_CALUDE_total_vehicles_l1895_189578

-- Define the number of trucks
def num_trucks : ℕ := 20

-- Define the number of tanks as a function of the number of trucks
def num_tanks : ℕ := 5 * num_trucks

-- Theorem to prove
theorem total_vehicles : num_tanks + num_trucks = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_l1895_189578


namespace NUMINAMATH_CALUDE_polynomial_roots_l1895_189527

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 13*x - 15

-- State the theorem
theorem polynomial_roots :
  (∃ a b c : ℝ, a < 0 ∧ 0 < b ∧ 0 < c ∧
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1895_189527


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l1895_189580

theorem sphere_surface_area_from_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) : 
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  4 * Real.pi * radius^2 = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l1895_189580


namespace NUMINAMATH_CALUDE_locus_is_two_ellipses_l1895_189585

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the locus of points
def LocusOfPoints (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs (dist p c1.center - c1.radius) = abs (dist p c2.center - c2.radius)}

-- Define the ellipse
def Ellipse (f1 f2 : ℝ × ℝ) (major_axis : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p f1 + dist p f2 = major_axis}

-- Theorem statement
theorem locus_is_two_ellipses (c1 c2 : Circle) 
  (h1 : c1.radius > c2.radius) 
  (h2 : dist c1.center c2.center < c1.radius - c2.radius) :
  LocusOfPoints c1 c2 = 
    Ellipse c1.center c2.center (c1.radius + c2.radius) ∪
    Ellipse c1.center c2.center (c1.radius - c2.radius) := by
  sorry


end NUMINAMATH_CALUDE_locus_is_two_ellipses_l1895_189585


namespace NUMINAMATH_CALUDE_smallest_number_is_2544_l1895_189595

def is_smallest_number (x : ℕ) : Prop :=
  (x - 24) % 5 = 0 ∧
  (x - 24) % 10 = 0 ∧
  (x - 24) % 15 = 0 ∧
  (x - 24) / (Nat.lcm 5 (Nat.lcm 10 15)) = 84 ∧
  ∀ y, y < x → ¬(is_smallest_number y)

theorem smallest_number_is_2544 :
  is_smallest_number 2544 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_is_2544_l1895_189595


namespace NUMINAMATH_CALUDE_solve_system_l1895_189573

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by sorry

end NUMINAMATH_CALUDE_solve_system_l1895_189573


namespace NUMINAMATH_CALUDE_frank_reading_speed_l1895_189562

/-- Given a book with a certain number of pages and the number of days to read it,
    calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that Frank read 102 pages per day. -/
theorem frank_reading_speed :
  pages_per_day 612 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_speed_l1895_189562


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l1895_189500

/-- Given points A(0, 0) and B(x, 5) where the slope of AB is 3/4,
    prove that the sum of x- and y-coordinates of B is 35/3 -/
theorem coordinate_sum_of_point_B (x : ℚ) : 
  let A : ℚ × ℚ := (0, 0)
  let B : ℚ × ℚ := (x, 5)
  let slope : ℚ := (B.2 - A.2) / (B.1 - A.1)
  slope = 3/4 → x + 5 = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l1895_189500


namespace NUMINAMATH_CALUDE_house_rent_expenditure_l1895_189535

/-- Given a person's income and expenditure pattern, calculate their house rent expense -/
theorem house_rent_expenditure (income : ℝ) (petrol_expense : ℝ) :
  petrol_expense = 0.3 * income →
  petrol_expense = 300 →
  let remaining_income := income - petrol_expense
  let house_rent := 0.2 * remaining_income
  house_rent = 140 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_expenditure_l1895_189535


namespace NUMINAMATH_CALUDE_original_cost_price_l1895_189598

/-- Calculates the original cost price given a series of transactions and the final price --/
theorem original_cost_price 
  (profit_ab profit_bc discount_cd profit_de final_price : ℝ) :
  let original_price := 
    final_price / ((1 + profit_ab/100) * (1 + profit_bc/100) * (1 - discount_cd/100) * (1 + profit_de/100))
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    (profit_ab = 20 ∧ 
     profit_bc = 25 ∧ 
     discount_cd = 15 ∧ 
     profit_de = 30 ∧ 
     final_price = 289.1) →
    (142.8 - ε ≤ original_price ∧ original_price ≤ 142.8 + ε) :=
by sorry

end NUMINAMATH_CALUDE_original_cost_price_l1895_189598


namespace NUMINAMATH_CALUDE_square_rectangle_area_ratio_l1895_189599

theorem square_rectangle_area_ratio :
  let rectangle_width : ℝ := 3
  let rectangle_length : ℝ := 5
  let square_side : ℝ := 1
  let square_area := square_side ^ 2
  let rectangle_area := rectangle_width * rectangle_length
  square_area / rectangle_area = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_area_ratio_l1895_189599


namespace NUMINAMATH_CALUDE_sum_of_digits_equals_four_l1895_189517

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5^i)) 0

/-- Converts a decimal number to base-6 --/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The base-5 representation of 2014₅ --/
def base5Number : List Nat := [4, 1, 0, 2]

theorem sum_of_digits_equals_four :
  let decimal := base5ToDecimal base5Number
  let base6 := decimalToBase6 decimal
  base6.sum = 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_equals_four_l1895_189517


namespace NUMINAMATH_CALUDE_square_of_binomial_l1895_189531

theorem square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16*x^2 + 40*x + a = (4*x + b)^2) → a = 25 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1895_189531


namespace NUMINAMATH_CALUDE_calculation_proof_l1895_189545

theorem calculation_proof : 
  71 * ((5 + 2/7) - (6 + 1/3)) / ((3 + 1/2) + (2 + 1/5)) = -(13 + 37/1197) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1895_189545


namespace NUMINAMATH_CALUDE_parabola_x_intercept_difference_l1895_189524

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x-value in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a quadratic function -/
def QuadraticFunction.contains_point (f : QuadraticFunction) (p : Point) : Prop :=
  f.eval p.x = p.y

/-- Calculates the difference between the x-intercepts of a quadratic function -/
noncomputable def x_intercept_difference (f : QuadraticFunction) : ℝ :=
  sorry

theorem parabola_x_intercept_difference :
  ∀ (f : QuadraticFunction),
  (∃ (v : Point), v.x = 3 ∧ v.y = -9 ∧ f.contains_point v) →
  f.contains_point ⟨5, 7⟩ →
  x_intercept_difference f = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercept_difference_l1895_189524


namespace NUMINAMATH_CALUDE_hair_length_calculation_l1895_189564

/-- Calculates the final hair length after a series of changes. -/
def finalHairLength (initialLength : ℝ) (firstCutFraction : ℝ) (growth : ℝ) (secondCut : ℝ) : ℝ :=
  (initialLength - firstCutFraction * initialLength + growth) - secondCut

/-- Theorem stating that given the specific hair length changes, the final length is 14 inches. -/
theorem hair_length_calculation :
  finalHairLength 24 0.5 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_hair_length_calculation_l1895_189564


namespace NUMINAMATH_CALUDE_video_game_lives_l1895_189559

theorem video_game_lives (initial lives_lost lives_gained : ℕ) :
  initial ≥ lives_lost →
  initial - lives_lost + lives_gained = initial + lives_gained - lives_lost :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l1895_189559


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1895_189542

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Given three collinear points (4, 10), (-3, k), and (-8, 5), prove that k = 85/12. -/
theorem collinear_points_k_value :
  collinear 4 10 (-3) k (-8) 5 → k = 85/12 :=
by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l1895_189542


namespace NUMINAMATH_CALUDE_sqrt_difference_comparison_l1895_189534

theorem sqrt_difference_comparison 
  (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hm : m > 0) 
  (hab : a > b) : 
  Real.sqrt (b + m) - Real.sqrt b > Real.sqrt (a + m) - Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_comparison_l1895_189534


namespace NUMINAMATH_CALUDE_recurrence_sequence_a8_l1895_189553

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem recurrence_sequence_a8 
  (a : ℕ → ℕ) 
  (h : RecurrenceSequence a) 
  (h7 : a 7 = 120) : 
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a8_l1895_189553


namespace NUMINAMATH_CALUDE_teachers_survey_l1895_189556

theorem teachers_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ) :
  total = 150 →
  high_bp = 90 →
  heart_trouble = 50 →
  both = 30 →
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 80 / 3 :=
by sorry

end NUMINAMATH_CALUDE_teachers_survey_l1895_189556


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1895_189540

theorem roots_of_quadratic (x : ℝ) : x^2 = 5*x ↔ x = 0 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1895_189540


namespace NUMINAMATH_CALUDE_group_size_calculation_l1895_189507

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 2.5 ∧ old_weight = 65 ∧ new_weight = 90 →
  ∃ n : ℕ, n = 10 ∧ n * average_increase = new_weight - old_weight :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l1895_189507


namespace NUMINAMATH_CALUDE_max_real_sum_l1895_189557

/-- The zeroes of z^10 - 2^30 -/
def zeroes : Finset ℂ :=
  sorry

/-- A function that chooses either z or iz to maximize the real part -/
def w (z : ℂ) : ℂ :=
  sorry

/-- The sum of w(z) for all zeroes -/
def sum_w : ℂ :=
  sorry

/-- The maximum possible value of the real part of the sum -/
theorem max_real_sum :
  (sum_w.re : ℝ) = 16 * (1 + Real.cos (π / 5) + Real.cos (2 * π / 5) - Real.sin (3 * π / 5) - Real.sin (4 * π / 5)) :=
sorry

end NUMINAMATH_CALUDE_max_real_sum_l1895_189557


namespace NUMINAMATH_CALUDE_percentage_problem_l1895_189508

theorem percentage_problem : ∃ p : ℝ, p > 0 ∧ p < 100 ∧ (p / 100) * 30 = (25 / 100) * 16 + 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1895_189508


namespace NUMINAMATH_CALUDE_pythagorean_sum_and_difference_squares_l1895_189550

theorem pythagorean_sum_and_difference_squares (a b c : ℕ+) 
  (h : c^2 = a^2 + b^2) : 
  ∃ (x y z w : ℕ+), c^2 + a*b = x^2 + y^2 ∧ c^2 - a*b = z^2 + w^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_sum_and_difference_squares_l1895_189550


namespace NUMINAMATH_CALUDE_factorial_division_l1895_189532

theorem factorial_division : 
  (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1895_189532


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1895_189516

theorem sin_330_degrees : Real.sin (330 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1895_189516


namespace NUMINAMATH_CALUDE_freddy_is_18_l1895_189512

def job_age : ℕ := 5

def stephanie_age (j : ℕ) : ℕ := 4 * j

def freddy_age (s : ℕ) : ℕ := s - 2

theorem freddy_is_18 : freddy_age (stephanie_age job_age) = 18 := by
  sorry

end NUMINAMATH_CALUDE_freddy_is_18_l1895_189512


namespace NUMINAMATH_CALUDE_total_amount_paid_l1895_189504

def grape_quantity : ℕ := 8
def grape_price : ℚ := 70
def mango_quantity : ℕ := 8
def mango_price : ℚ := 55
def orange_quantity : ℕ := 5
def orange_price : ℚ := 40
def apple_quantity : ℕ := 10
def apple_price : ℚ := 30
def grape_discount : ℚ := 0.1
def mango_tax : ℚ := 0.05

theorem total_amount_paid : 
  (grape_quantity * grape_price * (1 - grape_discount) + 
   mango_quantity * mango_price * (1 + mango_tax) + 
   orange_quantity * orange_price + 
   apple_quantity * apple_price) = 1466 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1895_189504


namespace NUMINAMATH_CALUDE_two_digit_average_decimal_l1895_189514

theorem two_digit_average_decimal (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 →  -- m and n are 2-digit positive integers
  (m + n) / 2 = m + n / 100 →            -- their average equals the decimal m.n
  min m n = 49 :=                        -- the smaller of m and n is 49
by sorry

end NUMINAMATH_CALUDE_two_digit_average_decimal_l1895_189514


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1895_189529

theorem opposite_of_negative_fraction :
  -(-(1 : ℚ) / 2023) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1895_189529


namespace NUMINAMATH_CALUDE_median_salary_is_24000_l1895_189570

structure SalaryGroup where
  title : String
  count : Nat
  salary : Nat

def company_data : List SalaryGroup := [
  ⟨"President", 1, 140000⟩,
  ⟨"Vice-President", 4, 92000⟩,
  ⟨"Director", 12, 75000⟩,
  ⟨"Associate Director", 8, 55000⟩,
  ⟨"Administrative Specialist", 38, 24000⟩
]

def total_employees : Nat := (company_data.map (λ g => g.count)).sum

theorem median_salary_is_24000 :
  total_employees = 63 →
  (∃ median_index : Nat, median_index = (total_employees + 1) / 2) →
  (∃ median_salary : Nat, 
    (company_data.map (λ g => List.replicate g.count g.salary)).join.get! (median_index - 1) = median_salary ∧
    median_salary = 24000) :=
by sorry

end NUMINAMATH_CALUDE_median_salary_is_24000_l1895_189570


namespace NUMINAMATH_CALUDE_function_correspondence_l1895_189530

-- Case 1
def A1 : Set ℕ := {1, 2, 3}
def B1 : Set ℕ := {7, 8, 9}
def f1 : ℕ → ℕ
  | 1 => 7
  | 2 => 7
  | 3 => 8
  | _ => 0  -- default case for completeness

-- Case 2
def A2 : Set ℕ := {1, 2, 3}
def B2 : Set ℕ := {1, 2, 3}
def f2 : ℕ → ℕ
  | x => 2 * x - 1

-- Case 3
def A3 : Set ℝ := {x : ℝ | x ≥ -1}
def B3 : Set ℝ := {x : ℝ | x ≥ -1}
def f3 : ℝ → ℝ
  | x => 2 * x + 1

-- Case 4
def A4 : Set ℤ := Set.univ
def B4 : Set ℤ := {-1, 1}
def f4 : ℤ → ℤ
  | n => if n % 2 = 0 then 1 else -1

theorem function_correspondence :
  (∀ x ∈ A1, f1 x ∈ B1) ∧
  (¬∀ x ∈ A2, f2 x ∈ B2) ∧
  (∀ x ∈ A3, f3 x ∈ B3) ∧
  (∀ x ∈ A4, f4 x ∈ B4) :=
by sorry

end NUMINAMATH_CALUDE_function_correspondence_l1895_189530


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1895_189582

theorem algebraic_expression_value (a b c : ℝ) : 
  (∀ x, (x - 1) * (x + 2) = a * x^2 + b * x + c) → 
  4 * a - 2 * b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1895_189582


namespace NUMINAMATH_CALUDE_cube_root_nested_expression_l1895_189594

theorem cube_root_nested_expression : 
  (2 * (2 * 8^(1/3))^(1/3))^(1/3) = 2^(5/9) := by sorry

end NUMINAMATH_CALUDE_cube_root_nested_expression_l1895_189594


namespace NUMINAMATH_CALUDE_symmetry_about_a_periodicity_l1895_189565

variable (f : ℝ → ℝ)
variable (a b : ℝ)

axiom a_nonzero : a ≠ 0
axiom b_diff_a : b ≠ a
axiom f_symmetry : ∀ x, f (a + x) = f (a - x)

theorem symmetry_about_a : ∀ x, f x = f (2*a - x) := by sorry

axiom symmetry_about_b : ∀ x, f x = f (2*b - x)

theorem periodicity : ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x := by sorry

end NUMINAMATH_CALUDE_symmetry_about_a_periodicity_l1895_189565


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1895_189558

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 5) :
  Real.tan α = -27/14 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1895_189558


namespace NUMINAMATH_CALUDE_sam_yellow_marbles_l1895_189568

/-- The number of yellow marbles Sam has after receiving more from Joan -/
def total_yellow_marbles (initial : Real) (received : Real) : Real :=
  initial + received

/-- Theorem stating that Sam now has 111.0 yellow marbles -/
theorem sam_yellow_marbles :
  total_yellow_marbles 86.0 25.0 = 111.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_yellow_marbles_l1895_189568


namespace NUMINAMATH_CALUDE_square_area_error_l1895_189590

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x * (1 + 0.17)
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.3689 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1895_189590


namespace NUMINAMATH_CALUDE_line_through_point_l1895_189576

/-- Theorem: If the line ax + 3y - 2 = 0 passes through point (1, 0), then a = 2. -/
theorem line_through_point (a : ℝ) : 
  (∀ x y, a * x + 3 * y - 2 = 0 → (x = 1 ∧ y = 0)) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_l1895_189576


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1895_189506

/-- The line 5x + 12y + a = 0 is tangent to the circle (x-1)^2 + y^2 = 1 if and only if a = -18 or a = 8 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, 5*x + 12*y + a = 0 → (x-1)^2 + y^2 = 1) ↔ (a = -18 ∨ a = 8) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1895_189506


namespace NUMINAMATH_CALUDE_max_intersections_four_circles_l1895_189552

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of intersections between a line and a circle --/
def intersectionCount (l : Line) (c : Circle) : ℕ := sorry

/-- Checks if four circles are coplanar --/
def areCoplanar (c1 c2 c3 c4 : Circle) : Prop := sorry

/-- Theorem: The maximum number of intersections between a line and four coplanar circles is 8 --/
theorem max_intersections_four_circles (c1 c2 c3 c4 : Circle) (l : Line) :
  areCoplanar c1 c2 c3 c4 →
  (intersectionCount l c1 + intersectionCount l c2 + intersectionCount l c3 + intersectionCount l c4) ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_four_circles_l1895_189552


namespace NUMINAMATH_CALUDE_g_symmetric_about_one_l1895_189549

-- Define the real-valued functions f and g
variable (f : ℝ → ℝ)
def g (x : ℝ) : ℝ := f (|x - 1|)

-- Define symmetry about a vertical line
def symmetric_about_line (h : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, h (a + x) = h (a - x)

-- Theorem statement
theorem g_symmetric_about_one (f : ℝ → ℝ) :
  symmetric_about_line (g f) 1 := by
  sorry

end NUMINAMATH_CALUDE_g_symmetric_about_one_l1895_189549


namespace NUMINAMATH_CALUDE_integer_factorization_l1895_189593

theorem integer_factorization (a b c d : ℤ) (h : a * b = c * d) :
  ∃ (w x y z : ℤ), a = w * x ∧ b = y * z ∧ c = w * y ∧ d = x * z := by
  sorry

end NUMINAMATH_CALUDE_integer_factorization_l1895_189593


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l1895_189523

-- Define a function to normalize an angle to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Theorem statement
theorem same_terminal_side_angle :
  normalizeAngle (-390) = 330 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l1895_189523


namespace NUMINAMATH_CALUDE_sequence_properties_l1895_189581

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (is_arithmetic_sequence a ∧ 
   (is_geometric_sequence (a 4) (a 7) (a 9) → 
    ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1895_189581


namespace NUMINAMATH_CALUDE_range_of_sum_l1895_189555

theorem range_of_sum (a b : ℝ) :
  (∀ x : ℝ, |x - a| + |x + b| ≥ 3) →
  a + b ∈ Set.Iic (-3) ∪ Set.Ioi 3 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l1895_189555


namespace NUMINAMATH_CALUDE_new_lines_satisfy_axioms_l1895_189539

-- Define the type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the type for new lines (parabolas and vertical lines)
inductive NewLine
  | Parabola (a b : ℝ)  -- y = (x + a)² + b
  | VerticalLine (c : ℝ)  -- x = c

-- Define when a point lies on a new line
def lies_on (p : Point) (l : NewLine) : Prop :=
  match l with
  | NewLine.Parabola a b => p.y = (p.x + a)^2 + b
  | NewLine.VerticalLine c => p.x = c

-- Axiom 1: For any two distinct points, there exists a unique new line passing through them
axiom exists_unique_newline (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! l : NewLine, lies_on p1 l ∧ lies_on p2 l

-- Axiom 2: Any two distinct new lines intersect in at most one point
axiom at_most_one_intersection (l1 l2 : NewLine) (h : l1 ≠ l2) :
  ∃! p : Point, lies_on p l1 ∧ lies_on p l2

-- Axiom 3: For any new line and a point not on it, there exists a unique new line
--          passing through the point and not intersecting the given line
axiom exists_unique_parallel (l : NewLine) (p : Point) (h : ¬lies_on p l) :
  ∃! l' : NewLine, lies_on p l' ∧ ∀ q : Point, ¬(lies_on q l ∧ lies_on q l')

-- Theorem: The set of new lines satisfies the three axioms
theorem new_lines_satisfy_axioms :
  (∀ p1 p2 : Point, p1 ≠ p2 → ∃! l : NewLine, lies_on p1 l ∧ lies_on p2 l) ∧
  (∀ l1 l2 : NewLine, l1 ≠ l2 → ∃! p : Point, lies_on p l1 ∧ lies_on p l2) ∧
  (∀ l : NewLine, ∀ p : Point, ¬lies_on p l →
    ∃! l' : NewLine, lies_on p l' ∧ ∀ q : Point, ¬(lies_on q l ∧ lies_on q l')) :=
by sorry

end NUMINAMATH_CALUDE_new_lines_satisfy_axioms_l1895_189539


namespace NUMINAMATH_CALUDE_rabbit_weight_l1895_189574

/-- Given the weights of a rabbit and two guinea pigs satisfying certain conditions,
    prove that the rabbit weighs 5 pounds. -/
theorem rabbit_weight (a b c : ℝ) 
  (total_weight : a + b + c = 30)
  (larger_smaller : a + c = 2 * b)
  (rabbit_smaller : a + b = c) : 
  a = 5 := by sorry

end NUMINAMATH_CALUDE_rabbit_weight_l1895_189574


namespace NUMINAMATH_CALUDE_second_worker_de_time_l1895_189551

/-- Represents a worker paving a path -/
structure Worker where
  speed : ℝ
  distance : ℝ

/-- Represents the paving scenario -/
structure PavingScenario where
  worker1 : Worker
  worker2 : Worker
  totalTime : ℝ

/-- The theorem statement -/
theorem second_worker_de_time (scenario : PavingScenario) : 
  scenario.worker1.speed > 0 ∧ 
  scenario.worker2.speed = 1.2 * scenario.worker1.speed ∧
  scenario.totalTime = 9 ∧
  scenario.worker1.distance * scenario.worker2.speed = scenario.worker2.distance * scenario.worker1.speed →
  ∃ (de_time : ℝ), de_time = 45 ∧ de_time = (scenario.totalTime * 60) / 12 :=
by sorry

end NUMINAMATH_CALUDE_second_worker_de_time_l1895_189551


namespace NUMINAMATH_CALUDE_inequality_proof_l1895_189586

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 2 * Real.sqrt 2 - 2) 
  (hb : b = Real.exp 2 / 7) 
  (hc : c = Real.log 2) : 
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1895_189586


namespace NUMINAMATH_CALUDE_unique_player_count_l1895_189587

/-- Given a total number of socks and the fact that each player contributes two socks,
    proves that there is only one possible number of players. -/
theorem unique_player_count (total_socks : ℕ) (h : total_socks = 22) :
  ∃! n : ℕ, n * 2 = total_socks := by sorry

end NUMINAMATH_CALUDE_unique_player_count_l1895_189587


namespace NUMINAMATH_CALUDE_napkin_ratio_l1895_189503

/-- Proves the ratio of napkins Amelia gave to napkins Olivia gave -/
theorem napkin_ratio (william_initial : ℕ) (william_final : ℕ) (olivia_gave : ℕ) 
  (h1 : william_initial = 15)
  (h2 : william_final = 45)
  (h3 : olivia_gave = 10) :
  (william_final - william_initial - olivia_gave) / olivia_gave = 2 := by
  sorry

end NUMINAMATH_CALUDE_napkin_ratio_l1895_189503


namespace NUMINAMATH_CALUDE_helen_lawn_mowing_gas_usage_l1895_189536

/-- Represents the lawn cutting schedule and gas usage for Helen's lawn mowing --/
structure LawnCuttingSchedule where
  march_to_october_low_freq : Nat  -- Number of months with 2 cuts per month
  may_to_august_high_freq : Nat    -- Number of months with 4 cuts per month
  cuts_per_low_freq_month : Nat    -- Number of cuts in low frequency months
  cuts_per_high_freq_month : Nat   -- Number of cuts in high frequency months
  gas_usage_frequency : Nat        -- Every nth cut uses gas
  gas_usage_amount : Nat           -- Amount of gas used every nth cut

/-- Calculates the total gas usage for Helen's lawn mowing schedule --/
def calculate_gas_usage (schedule : LawnCuttingSchedule) : Nat :=
  let total_cuts := 
    schedule.march_to_october_low_freq * schedule.cuts_per_low_freq_month +
    schedule.may_to_august_high_freq * schedule.cuts_per_high_freq_month
  let gas_usage_times := total_cuts / schedule.gas_usage_frequency
  gas_usage_times * schedule.gas_usage_amount

/-- Theorem stating that Helen's lawn mowing schedule results in 12 gallons of gas usage --/
theorem helen_lawn_mowing_gas_usage :
  let schedule : LawnCuttingSchedule := {
    march_to_october_low_freq := 4
    may_to_august_high_freq := 4
    cuts_per_low_freq_month := 2
    cuts_per_high_freq_month := 4
    gas_usage_frequency := 4
    gas_usage_amount := 2
  }
  calculate_gas_usage schedule = 12 := by
  sorry


end NUMINAMATH_CALUDE_helen_lawn_mowing_gas_usage_l1895_189536


namespace NUMINAMATH_CALUDE_cos_a_minus_pi_l1895_189543

theorem cos_a_minus_pi (a : Real) 
  (h1 : π / 2 < a ∧ a < π) 
  (h2 : 3 * Real.sin (2 * a) = 2 * Real.cos a) : 
  Real.cos (a - π) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_minus_pi_l1895_189543


namespace NUMINAMATH_CALUDE_sin_ratio_comparison_l1895_189520

theorem sin_ratio_comparison : (Real.sin (3 * π / 180)) / (Real.sin (4 * π / 180)) > (Real.sin (1 * π / 180)) / (Real.sin (2 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_sin_ratio_comparison_l1895_189520


namespace NUMINAMATH_CALUDE_P_divisible_by_factor_l1895_189584

def P (x : ℝ) : ℝ := (x + 1)^6 - x^6 - 2*x - 1

theorem P_divisible_by_factor : ∃ Q : ℝ → ℝ, ∀ x : ℝ, P x = x * (x + 1) * (2*x + 1) * Q x := by
  sorry

end NUMINAMATH_CALUDE_P_divisible_by_factor_l1895_189584


namespace NUMINAMATH_CALUDE_cube_root_problem_l1895_189521

theorem cube_root_problem :
  ∀ (a b : ℤ) (c : ℚ),
  (5 * a - 2 : ℚ) = -27 →
  b = ⌊Real.sqrt 22⌋ →
  c = -(4 / 25 : ℚ).sqrt →
  a = -5 ∧
  b = 4 ∧
  c = -2/5 ∧
  Real.sqrt (4 * (a : ℚ) * c + 7 * (b : ℚ)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1895_189521


namespace NUMINAMATH_CALUDE_problem_4_l1895_189519

theorem problem_4 (a : ℝ) : (2*a + 1)^2 - (2*a + 1)*(2*a - 1) = 4*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_4_l1895_189519


namespace NUMINAMATH_CALUDE_debby_ate_nine_candies_l1895_189588

/-- Represents the number of candy pieces Debby ate -/
def candy_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proves that Debby ate 9 pieces of candy -/
theorem debby_ate_nine_candies : candy_eaten 12 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_debby_ate_nine_candies_l1895_189588


namespace NUMINAMATH_CALUDE_janets_group_children_count_l1895_189554

theorem janets_group_children_count 
  (total_people : Nat) 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (discount_rate : ℚ) 
  (soda_price : ℚ) 
  (total_paid : ℚ) :
  total_people = 10 ∧ 
  adult_price = 30 ∧ 
  child_price = 15 ∧ 
  discount_rate = 0.8 ∧ 
  soda_price = 5 ∧ 
  total_paid = 197 →
  ∃ (children : Nat),
    children ≤ total_people ∧
    (total_paid - soda_price) = 
      ((adult_price * (total_people - children) + child_price * children) * discount_rate) ∧
    children = 4 := by
  sorry

end NUMINAMATH_CALUDE_janets_group_children_count_l1895_189554


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l1895_189501

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The length of the shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- The length of the longer segment of the non-parallel side divided by the point of tangency -/
  longerSegment : ℝ
  /-- The length of the shorter segment of the non-parallel side divided by the point of tangency -/
  shorterSegment : ℝ
  /-- The shorter base is positive -/
  shorterBase_pos : 0 < shorterBase
  /-- The longer segment is positive -/
  longerSegment_pos : 0 < longerSegment
  /-- The shorter segment is positive -/
  shorterSegment_pos : 0 < shorterSegment

/-- The area of the trapezoid -/
def area (t : InscribedCircleTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the specific trapezoid is 198 -/
theorem specific_trapezoid_area :
  ∀ t : InscribedCircleTrapezoid,
  t.shorterBase = 6 ∧ t.longerSegment = 9 ∧ t.shorterSegment = 4 →
  area t = 198 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l1895_189501


namespace NUMINAMATH_CALUDE_magic_square_x_value_l1895_189548

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  sum_eq : a + b + c = d + e + f ∧ 
           a + b + c = g + h + i ∧ 
           a + b + c = a + d + g ∧ 
           a + b + c = b + e + h ∧ 
           a + b + c = c + f + i ∧ 
           a + b + c = a + e + i ∧ 
           a + b + c = c + e + g

/-- Theorem stating that x must be 230 in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a = x)
  (h2 : ms.b = 25)
  (h3 : ms.c = 110)
  (h4 : ms.d = 5) :
  x = 230 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l1895_189548


namespace NUMINAMATH_CALUDE_system_solution_l1895_189502

theorem system_solution : 
  ∃ (x y : ℚ), 2 * x + 3 * y = 1 ∧ 3 * x - 6 * y = 7 ∧ x = 9/7 ∧ y = -11/21 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1895_189502


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1895_189596

/-- The distance between the foci of the ellipse x^2 + 9y^2 = 324 is 24√2 -/
theorem ellipse_foci_distance : 
  let ellipse_equation := fun (x y : ℝ) => x^2 + 9*y^2 = 324
  ∃ f₁ f₂ : ℝ × ℝ, 
    (∀ x y, ellipse_equation x y → ((x - f₁.1)^2 + (y - f₁.2)^2) + ((x - f₂.1)^2 + (y - f₂.2)^2) = 2 * 324) ∧ 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (24 * Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1895_189596


namespace NUMINAMATH_CALUDE_bowling_ball_weighs_16_pounds_l1895_189592

/-- The weight of a single bowling ball in pounds. -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of a single canoe in pounds. -/
def canoe_weight : ℝ := sorry

/-- Theorem stating that a bowling ball weighs 16 pounds under given conditions. -/
theorem bowling_ball_weighs_16_pounds : bowling_ball_weight = 16 := by
  have h1 : 8 * bowling_ball_weight = 4 * canoe_weight := by sorry
  have h2 : 2 * canoe_weight = 64 := by sorry
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weighs_16_pounds_l1895_189592


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l1895_189533

/-- A quadratic function with a negative leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_neg : a < 0

/-- The function f(x) -/
def f (qf : QuadraticFunction) (x : ℝ) : ℝ := qf.a * x^2 + qf.b * x + qf.c

/-- The condition that 1 and 3 are roots of y = f(x) + 2x -/
def roots_condition (qf : QuadraticFunction) : Prop :=
  f qf 1 + 2 * 1 = 0 ∧ f qf 3 + 2 * 3 = 0

/-- The condition that f(x) + 6a = 0 has two equal roots -/
def equal_roots_condition (qf : QuadraticFunction) : Prop :=
  ∃ (x : ℝ), f qf x + 6 * qf.a = 0 ∧ 
  ∀ (y : ℝ), f qf y + 6 * qf.a = 0 → y = x

/-- The theorem statement -/
theorem unique_quadratic_function :
  ∃! (qf : QuadraticFunction),
    roots_condition qf ∧
    equal_roots_condition qf ∧
    qf.a = -1/4 ∧ qf.b = -1 ∧ qf.c = -3/4 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l1895_189533


namespace NUMINAMATH_CALUDE_octal_subtraction_example_l1895_189546

/-- Represents a number in base 8 as a list of digits (least significant first) --/
def OctalNumber := List Nat

/-- Subtraction operation for octal numbers --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from a natural number to its octal representation --/
def to_octal (n : Nat) : OctalNumber :=
  sorry

theorem octal_subtraction_example :
  octal_subtract [4, 3, 5, 7] [7, 6, 2, 3] = [3, 4, 2, 4] :=
sorry

end NUMINAMATH_CALUDE_octal_subtraction_example_l1895_189546


namespace NUMINAMATH_CALUDE_x_value_l1895_189537

/-- Given that ( √x ) / ( √0.81 ) + ( √1.44 ) / ( √0.49 ) = 2.879628878919216, prove that x = 1.1 -/
theorem x_value (x : ℝ) 
  (h : Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49 = 2.879628878919216) : 
  x = 1.1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1895_189537


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1895_189525

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1895_189525


namespace NUMINAMATH_CALUDE_sector_central_angle_l1895_189569

theorem sector_central_angle (s : Real) (r : Real) (θ : Real) 
  (h1 : s = π) 
  (h2 : r = 2) 
  (h3 : s = r * θ) : θ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1895_189569


namespace NUMINAMATH_CALUDE_largest_common_term_l1895_189515

def first_sequence (n : ℕ) : ℕ := 3 + 10 * (n - 1)
def second_sequence (n : ℕ) : ℕ := 5 + 8 * (n - 1)

theorem largest_common_term : 
  (∃ (n m : ℕ), first_sequence n = second_sequence m ∧ first_sequence n = 133) ∧
  (∀ (x : ℕ), x > 133 → x ≤ 150 → 
    (∀ (n m : ℕ), first_sequence n ≠ x ∨ second_sequence m ≠ x)) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l1895_189515


namespace NUMINAMATH_CALUDE_square_between_bounds_l1895_189541

theorem square_between_bounds (n : ℕ) (hn : n ≥ 16088121) :
  ∃ l : ℕ, n < l ^ 2 ∧ l ^ 2 < n * (1 + 1 / 2005) := by
  sorry

end NUMINAMATH_CALUDE_square_between_bounds_l1895_189541


namespace NUMINAMATH_CALUDE_servant_served_nine_months_l1895_189577

/-- Represents the employment contract and service details of a servant -/
structure ServantContract where
  fullYearSalary : ℕ  -- Salary for a full year in rupees
  uniformPrice : ℕ    -- Price of the uniform in rupees
  receivedSalary : ℕ  -- Salary actually received in rupees
  fullYearMonths : ℕ  -- Number of months in a full year

/-- Calculates the number of months served by the servant -/
def monthsServed (contract : ServantContract) : ℕ :=
  (contract.receivedSalary + contract.uniformPrice) * contract.fullYearMonths 
    / (contract.fullYearSalary + contract.uniformPrice)

/-- Theorem stating that the servant served for 9 months -/
theorem servant_served_nine_months :
  let contract : ServantContract := {
    fullYearSalary := 500,
    uniformPrice := 500,
    receivedSalary := 250,
    fullYearMonths := 12
  }
  monthsServed contract = 9 := by sorry

end NUMINAMATH_CALUDE_servant_served_nine_months_l1895_189577


namespace NUMINAMATH_CALUDE_square_difference_div_product_equals_four_l1895_189591

theorem square_difference_div_product_equals_four :
  ((0.137 + 0.098)^2 - (0.137 - 0.098)^2) / (0.137 * 0.098) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_div_product_equals_four_l1895_189591


namespace NUMINAMATH_CALUDE_average_score_is_490_l1895_189566

-- Define the maximum score
def max_score : ℕ := 700

-- Define the number of students
def num_students : ℕ := 4

-- Define the scores as percentages
def gibi_percent : ℕ := 59
def jigi_percent : ℕ := 55
def mike_percent : ℕ := 99
def lizzy_percent : ℕ := 67

-- Define a function to calculate the actual score from a percentage
def calculate_score (percent : ℕ) : ℕ :=
  (percent * max_score) / 100

-- Theorem to prove
theorem average_score_is_490 : 
  (calculate_score gibi_percent + calculate_score jigi_percent + 
   calculate_score mike_percent + calculate_score lizzy_percent) / num_students = 490 :=
by sorry

end NUMINAMATH_CALUDE_average_score_is_490_l1895_189566


namespace NUMINAMATH_CALUDE_car_travel_time_l1895_189513

/-- Proves that a car traveling 810 km at 162 km/h takes 5 hours -/
theorem car_travel_time (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 810 ∧ speed = 162 → time = distance / speed → time = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l1895_189513


namespace NUMINAMATH_CALUDE_pizza_varieties_count_l1895_189572

/-- Represents the number of base pizza flavors -/
def num_flavors : ℕ := 8

/-- Represents the number of extra topping options -/
def num_toppings : ℕ := 5

/-- Calculates the number of valid topping combinations -/
def valid_topping_combinations : ℕ :=
  (num_toppings) +  -- 1 topping
  (num_toppings.choose 2 - 1) +  -- 2 toppings, excluding onions with jalapeños
  (num_toppings.choose 3 - 3)  -- 3 toppings, excluding combinations with both onions and jalapeños

/-- The total number of pizza varieties -/
def total_varieties : ℕ := num_flavors * valid_topping_combinations

theorem pizza_varieties_count :
  total_varieties = 168 := by sorry

end NUMINAMATH_CALUDE_pizza_varieties_count_l1895_189572


namespace NUMINAMATH_CALUDE_correct_propositions_count_l1895_189544

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def skew_lines (l1 l2 : Line) : Prop := sorry

-- Define the propositions
def proposition1 (m n : Line) (α : Plane) : Prop :=
  perpendicular_lines m n → perpendicular m α → parallel n α

def proposition2 (m n : Line) (α β : Plane) : Prop :=
  perpendicular m α → perpendicular n β → parallel_lines m n → parallel_planes α β

def proposition3 (m n : Line) (α β : Plane) : Prop :=
  skew_lines m n → line_in_plane m α → line_in_plane n β → parallel m β → parallel n α → parallel_planes α β

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  perpendicular_planes α β → intersection α β = m → line_in_plane n β → perpendicular_lines n m → perpendicular n α

-- Theorem statement
theorem correct_propositions_count :
  ¬proposition1 m n α ∧
  proposition2 m n α β ∧
  proposition3 m n α β ∧
  proposition4 m n α β :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_count_l1895_189544


namespace NUMINAMATH_CALUDE_triangle_obtuse_iff_tangent_product_less_than_one_l1895_189561

theorem triangle_obtuse_iff_tangent_product_less_than_one 
  (α β γ : Real) (h_sum : α + β + γ = Real.pi) (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) :
  γ > Real.pi / 2 ↔ Real.tan α * Real.tan β < 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_obtuse_iff_tangent_product_less_than_one_l1895_189561


namespace NUMINAMATH_CALUDE_connie_marbles_l1895_189547

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l1895_189547
