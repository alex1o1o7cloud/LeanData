import Mathlib

namespace max_sum_of_factors_l699_69965

theorem max_sum_of_factors (P Q R : ℕ+) : 
  P ≠ Q → P ≠ R → Q ≠ R → P * Q * R = 5103 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → X ≠ Z → Y ≠ Z → X * Y * Z = 5103 → 
  P + Q + R ≤ 136 ∧ (∃ (A B C : ℕ+), A ≠ B → A ≠ C → B ≠ C → A * B * C = 5103 ∧ A + B + C = 136) := by
sorry

end max_sum_of_factors_l699_69965


namespace square_paper_side_length_l699_69959

/-- The length of a cube's edge in centimeters -/
def cube_edge : ℝ := 12

/-- The number of square paper pieces covering the cube -/
def num_squares : ℕ := 54

/-- The surface area of a cube given its edge length -/
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

/-- The theorem stating that the side length of each square paper is 4 cm -/
theorem square_paper_side_length :
  ∃ (side : ℝ),
    side > 0 ∧
    side^2 * num_squares = cube_surface_area cube_edge ∧
    side = 4 := by
  sorry

end square_paper_side_length_l699_69959


namespace rectangle_width_l699_69946

/-- Given a rectangle and a triangle, if the ratio of their areas is 2:5,
    the rectangle has length 6 cm, and the triangle has area 60 cm²,
    then the width of the rectangle is 4 cm. -/
theorem rectangle_width (length width : ℝ) (triangle_area : ℝ) : 
  length = 6 →
  triangle_area = 60 →
  (length * width) / triangle_area = 2 / 5 →
  width = 4 := by
  sorry

end rectangle_width_l699_69946


namespace inverse_function_solution_l699_69973

/-- Given a function f(x) = 1 / (2ax + 3b) where a and b are nonzero constants,
    prove that the solution to f⁻¹(x) = -1 is x = 1 / (-2a + 3b) -/
theorem inverse_function_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => 1 / (2 * a * x + 3 * b)
  ∃! x, f x = -1 ∧ x = 1 / (-2 * a + 3 * b) := by
  sorry

end inverse_function_solution_l699_69973


namespace cubic_function_properties_l699_69954

/-- A cubic function with specific properties -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x

/-- The derivative of the cubic function -/
def f_deriv (p q : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*p*x + q

theorem cubic_function_properties (p q : ℝ) :
  (∃ a : ℝ, a ≠ 0 ∧ f p q a = 0) →  -- Intersects x-axis at non-origin point
  (∃ x_min : ℝ, f p q x_min = -4 ∧ ∀ x : ℝ, f p q x ≥ -4) →  -- Minimum y-value is -4
  p = 6 ∧ q = 9 := by
  sorry

end cubic_function_properties_l699_69954


namespace point_on_terminal_side_l699_69905

theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (3 : ℝ) / Real.sqrt ((3 : ℝ)^2 + m^2) = (3 : ℝ) / 5 → m = 4 ∨ m = -4 := by
  sorry

end point_on_terminal_side_l699_69905


namespace basketball_cricket_students_l699_69948

theorem basketball_cricket_students (basketball : ℕ) (cricket : ℕ) (both : ℕ)
  (h1 : basketball = 12)
  (h2 : cricket = 8)
  (h3 : both = 3) :
  basketball + cricket - both = 17 := by
sorry

end basketball_cricket_students_l699_69948


namespace min_triangles_to_cover_l699_69925

theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 2 → large_side = 12 → 
  (large_side / small_side) ^ 2 = 36 := by
  sorry

end min_triangles_to_cover_l699_69925


namespace curve_intersects_axes_l699_69922

-- Define the parametric equations
def curve_x (t : ℝ) : ℝ := t - 1
def curve_y (t : ℝ) : ℝ := t + 2

-- Define the curve as a set of points
def curve : Set (ℝ × ℝ) := {(x, y) | ∃ t : ℝ, x = curve_x t ∧ y = curve_y t}

-- Define the coordinate axes
def x_axis : Set (ℝ × ℝ) := {(x, y) | y = 0}
def y_axis : Set (ℝ × ℝ) := {(x, y) | x = 0}

theorem curve_intersects_axes :
  (0, 3) ∈ curve ∩ y_axis ∧ (-3, 0) ∈ curve ∩ x_axis :=
sorry

end curve_intersects_axes_l699_69922


namespace parliament_vote_ratio_l699_69962

theorem parliament_vote_ratio (V : ℝ) (X : ℝ) (q : ℝ) : 
  V > 0 →
  X = 7/10 * V →
  (6/50 * V + q * X) / (9/50 * V + (1 - q) * X) = 3/2 →
  q = 24/35 := by
  sorry

end parliament_vote_ratio_l699_69962


namespace triangle_BCD_properties_l699_69924

/-- Triangle BCD with given properties -/
structure TriangleBCD where
  BC : ℝ
  CD : ℝ
  M : ℝ × ℝ  -- Point M on BD
  angle_BCM : ℝ
  angle_MCD : ℝ
  h_BC : BC = 3
  h_CD : CD = 5
  h_angle_BCM : angle_BCM = π / 4  -- 45°
  h_angle_MCD : angle_MCD = π / 3  -- 60°

/-- Theorem about the properties of Triangle BCD -/
theorem triangle_BCD_properties (t : TriangleBCD) :
  -- 1. Ratio of BM to MD
  (∃ k : ℝ, k > 0 ∧ t.M.1 / t.M.2 = Real.sqrt 6 / 5 * k) ∧
  -- 2. Length of BM
  t.M.1 = (Real.sqrt 3 * Real.sqrt (68 + 15 * Real.sqrt 2 * (Real.sqrt 3 - 1))) / (Real.sqrt 6 + 5) ∧
  -- 3. Length of MD
  t.M.2 = (5 * Real.sqrt (68 + 15 * Real.sqrt 2 * (Real.sqrt 3 - 1))) / (2 * Real.sqrt 3 + 5 * Real.sqrt 2) :=
by sorry


end triangle_BCD_properties_l699_69924


namespace no_integer_solution_l699_69952

theorem no_integer_solution : 
  ¬∃ (x y : ℤ), (18 * x + 27 * y = 21) ∧ (27 * x + 18 * y = 69) := by
  sorry

end no_integer_solution_l699_69952


namespace remainder_theorem_l699_69927

def f (x : ℝ) : ℝ := x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_theorem : ∃ (Q : ℝ → ℝ), ∀ x, f (x^10) = f x * Q x + 10 := by
  sorry

end remainder_theorem_l699_69927


namespace quadratic_root_shift_l699_69958

theorem quadratic_root_shift (a b : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, x = 2021 ∧ a * x^2 + b * x + 2 = 0) →
  (∃ y : ℝ, y = 2022 ∧ a * (y - 1)^2 + b * (y - 1) = -2) :=
by sorry

end quadratic_root_shift_l699_69958


namespace smallest_digit_count_l699_69979

theorem smallest_digit_count (a n : ℕ) (h : (Nat.log 10 (a^n) + 1 = 2014)) :
  ∀ k < 2014, ∃ a' : ℕ, 10^(k-1) ≤ a' ∧ a' < 10^k ∧ (Nat.log 10 (a'^n) + 1 = 2014) ∧
  ¬∃ a' : ℕ, 10^2013 ≤ a' ∧ a' < 10^2014 ∧ (Nat.log 10 (a'^n) + 1 = 2014) :=
by sorry

end smallest_digit_count_l699_69979


namespace second_number_value_l699_69996

def average_of_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem second_number_value (x y : ℚ) 
  (h1 : average_of_three 2 y x = 5) 
  (h2 : x = -63) : 
  y = 76 := by
  sorry

end second_number_value_l699_69996


namespace quotient_problem_l699_69930

theorem quotient_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : 1 / (3 * a) + 1 / b = 2011)
  (h2 : 1 / a + 1 / (3 * b) = 1) :
  (a + b) / (a * b) = 1509 := by
  sorry

end quotient_problem_l699_69930


namespace inequality_proof_l699_69907

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 := by
  sorry

end inequality_proof_l699_69907


namespace min_value_of_sum_l699_69980

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ x + y ≥ m ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ x₀ + y₀ = m :=
by sorry

end min_value_of_sum_l699_69980


namespace decimal_fraction_equality_l699_69971

theorem decimal_fraction_equality : (0.2^3) / (0.02^2) = 20 := by sorry

end decimal_fraction_equality_l699_69971


namespace weather_forecast_probability_l699_69942

-- Define the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the binomial probability mass function
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem weather_forecast_probability :
  let n : ℕ := 3
  let p : ℝ := 0.8
  let k : ℕ := 2
  binomial_pmf n p k = 0.384 := by sorry

end weather_forecast_probability_l699_69942


namespace watch_correction_theorem_l699_69966

/-- Represents the number of days between two dates -/
def daysBetween (startDate endDate : Nat) : Nat :=
  endDate - startDate

/-- Represents the number of hours from noon to 10 AM the next day -/
def hoursFromNoonTo10AM : Nat := 22

/-- Represents the rate at which the watch loses time in minutes per day -/
def watchLossRate : Rat := 13/4

/-- Calculates the total hours elapsed on the watch -/
def totalWatchHours (days : Nat) : Nat :=
  days * 24 + hoursFromNoonTo10AM

/-- Calculates the total time loss in minutes -/
def totalTimeLoss (hours : Nat) (lossRate : Rat) : Rat :=
  (hours : Rat) * (lossRate / 24)

/-- The initial time difference when the watch was set -/
def initialTimeDifference : Rat := 10

/-- Theorem: The positive correction to be added to the watch time is 35.7292 minutes -/
theorem watch_correction_theorem (startDate endDate : Nat) :
  let days := daysBetween startDate endDate
  let hours := totalWatchHours days
  let loss := totalTimeLoss hours watchLossRate
  loss + initialTimeDifference = 35.7292 := by sorry

end watch_correction_theorem_l699_69966


namespace purely_imaginary_complex_number_l699_69915

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := (m - 1) * (m - 2) + (m - 2) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
  sorry

end purely_imaginary_complex_number_l699_69915


namespace tangent_line_speed_l699_69903

theorem tangent_line_speed 
  (a T R L x : ℝ) 
  (h_pos : a > 0 ∧ T > 0 ∧ R > 0 ∧ L > 0)
  (h_eq : (a * T) / (a * T - R) = (L + x) / x) :
  x / T = a * L / R :=
sorry

end tangent_line_speed_l699_69903


namespace power_of_four_exponent_l699_69951

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (hn : n = 21) : x = 22 := by
  sorry

end power_of_four_exponent_l699_69951


namespace tree_initial_height_l699_69970

/-- Represents the height of a tree over time -/
def TreeHeight (H : ℝ) (t : ℕ) : ℝ := H + t

/-- The problem statement about the tree's growth -/
theorem tree_initial_height :
  ∀ H : ℝ,
  (TreeHeight H 6 = TreeHeight H 4 + (1/4) * TreeHeight H 4) →
  H = 4 := by
  sorry

end tree_initial_height_l699_69970


namespace intersection_of_sets_l699_69986

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {0, 2, 3}
  A ∩ B = {2, 3} := by
sorry

end intersection_of_sets_l699_69986


namespace jeff_tennis_matches_l699_69910

/-- Calculates the number of matches won in a tennis game -/
def matches_won (total_time : ℕ) (point_interval : ℕ) (points_per_match : ℕ) : ℕ :=
  (total_time * 60 / point_interval) / points_per_match

/-- Theorem stating that under given conditions, 3 matches are won -/
theorem jeff_tennis_matches : 
  matches_won 2 5 8 = 3 := by
  sorry

end jeff_tennis_matches_l699_69910


namespace min_turns_10x10_grid_l699_69983

/-- Represents a grid of intersecting streets -/
structure StreetGrid where
  parallel_streets : ℕ
  intersecting_streets : ℕ

/-- Calculates the minimum number of turns required for a closed route
    passing through all intersections in a grid of streets -/
def min_turns (grid : StreetGrid) : ℕ :=
  2 * grid.parallel_streets

/-- The theorem stating that in a 10x10 grid of intersecting streets,
    the minimum number of turns required for a closed route passing
    through all intersections is 20 -/
theorem min_turns_10x10_grid :
  let grid : StreetGrid := ⟨10, 10⟩
  min_turns grid = 20 := by
  sorry

end min_turns_10x10_grid_l699_69983


namespace problem_statement_l699_69960

theorem problem_statement : 2006 * ((Real.sqrt 8 - Real.sqrt 2) / Real.sqrt 2) = 2006 := by
  sorry

end problem_statement_l699_69960


namespace roses_cut_correct_l699_69901

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Mary cut is correct -/
theorem roses_cut_correct (initial_roses final_roses : ℕ) 
  (h : initial_roses ≤ final_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16

end roses_cut_correct_l699_69901


namespace arithmetic_sequence_sum_l699_69957

theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 48 → S (2 * n) = 60 → S (3 * n) = 63 := by
  sorry

end arithmetic_sequence_sum_l699_69957


namespace total_clothes_ironed_l699_69967

/-- The time in minutes it takes Eliza to iron a blouse -/
def blouse_time : ℕ := 15

/-- The time in minutes it takes Eliza to iron a dress -/
def dress_time : ℕ := 20

/-- The time in hours Eliza spends ironing blouses -/
def blouse_hours : ℕ := 2

/-- The time in hours Eliza spends ironing dresses -/
def dress_hours : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the total number of pieces of clothes Eliza ironed -/
theorem total_clothes_ironed : 
  (blouse_hours * minutes_per_hour / blouse_time) + 
  (dress_hours * minutes_per_hour / dress_time) = 17 := by
  sorry

end total_clothes_ironed_l699_69967


namespace coin_difference_l699_69950

def coin_values : List ℕ := [10, 20, 50]
def target_amount : ℕ := 145

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.sum * 10 ≥ target_amount ∧
  (combination.sum * 10 - target_amount < 10 ∨ combination.sum * 10 = target_amount)

def min_coins : ℕ := sorry
def max_coins : ℕ := sorry

theorem coin_difference :
  ∃ (min_comb max_comb : List ℕ),
    is_valid_combination min_comb ∧
    is_valid_combination max_comb ∧
    min_comb.length = min_coins ∧
    max_comb.length = max_coins ∧
    max_coins - min_coins = 9 :=
  sorry

end coin_difference_l699_69950


namespace pastries_sold_l699_69919

/-- Given that a baker initially had 56 pastries and now has 27 pastries remaining,
    prove that the number of pastries sold is 29. -/
theorem pastries_sold (initial_pastries : ℕ) (remaining_pastries : ℕ) 
  (h1 : initial_pastries = 56) (h2 : remaining_pastries = 27) : 
  initial_pastries - remaining_pastries = 29 := by
  sorry

end pastries_sold_l699_69919


namespace andrews_payment_l699_69936

/-- Calculates the total amount paid for fruits with discounts and taxes -/
def totalAmountPaid (grapeQuantity grapePrice mangoQuantity mangoPrice appleQuantity applePrice orangeQuantity orangePrice discountRate taxRate : ℝ) : ℝ :=
  let grapeCost := grapeQuantity * grapePrice
  let mangoCost := mangoQuantity * mangoPrice
  let appleCost := appleQuantity * applePrice
  let orangeCost := orangeQuantity * orangePrice
  let grapeMangoCost := grapeCost + mangoCost
  let discountAmount := discountRate * grapeMangoCost
  let discountedGrapeMangoCost := grapeMangoCost - discountAmount
  let totalCostBeforeTax := discountedGrapeMangoCost + appleCost + orangeCost
  let taxAmount := taxRate * totalCostBeforeTax
  totalCostBeforeTax + taxAmount

/-- Theorem stating that Andrew's total payment is $1306.41 -/
theorem andrews_payment :
  totalAmountPaid 7 68 9 48 5 55 4 38 0.1 0.05 = 1306.41 := by
  sorry

end andrews_payment_l699_69936


namespace preimage_of_20_l699_69928

def A : Set ℕ := sorry
def B : Set ℕ := sorry

def f (n : ℕ) : ℕ := 2 * n^2

theorem preimage_of_20 : ∃ (n : ℕ), n ∈ A ∧ f n = 20 ∧ (∀ (m : ℕ), m ∈ A ∧ f m = 20 → m = n) :=
  sorry

end preimage_of_20_l699_69928


namespace tan_150_degrees_l699_69997

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l699_69997


namespace sqrt_of_four_l699_69982

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_four_l699_69982


namespace rug_overlap_problem_l699_69923

/-- Given three rugs with specific overlapping conditions, prove the area covered by exactly two layers -/
theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (triple_layer_area : ℝ) 
  (h1 : total_area = 204)
  (h2 : covered_area = 140)
  (h3 : triple_layer_area = 20) :
  total_area - 2 * triple_layer_area - covered_area = 24 := by
  sorry

end rug_overlap_problem_l699_69923


namespace product_representation_l699_69963

theorem product_representation (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  ∃ p q : ℝ, (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) = p + q * Real.sqrt 5 ∧ p ≥ 0 ∧ q ≥ 0 := by
  sorry

end product_representation_l699_69963


namespace factor_implies_abs_value_l699_69913

-- Define the polynomial
def p (x h k : ℚ) : ℚ := 3 * x^4 - 2 * h * x^2 - 5 * x + k

-- State the theorem
theorem factor_implies_abs_value (h k : ℚ) : 
  (∀ x, p x h k = 0 → x = 1 ∨ x = -3) → 
  |3 * h + 2 * k| = 471 / 4 := by
  sorry

end factor_implies_abs_value_l699_69913


namespace track_length_is_600_l699_69987

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- The conditions of the problem -/
def track_conditions (t : CircularTrack) : Prop :=
  ∃ (first_meeting second_meeting : ℝ),
    first_meeting > 0 ∧
    second_meeting > first_meeting ∧
    first_meeting * t.runner2_speed = 120 ∧
    (second_meeting - first_meeting) * t.runner1_speed = 180 ∧
    first_meeting * t.runner1_speed + 120 = t.length / 2 ∧
    t.runner1_speed > 0 ∧
    t.runner2_speed > 0

/-- The theorem to be proved -/
theorem track_length_is_600 (t : CircularTrack) :
  track_conditions t → t.length = 600 := by
  sorry

end track_length_is_600_l699_69987


namespace quadratic_a_value_l699_69939

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ
  vertex_form : ∀ x, a * (x - h)^2 + k = a * x^2 + ((-2 * a * h) * x) + (a * h^2 + k)
  passes_through : a * (x₀ - h)^2 + k = y₀

/-- The theorem stating that for a quadratic function with vertex (2, 5) 
    passing through (-1, -20), the value of 'a' is -25/9 -/
theorem quadratic_a_value (f : QuadraticFunction) 
    (vertex_h : f.h = 2) 
    (vertex_k : f.k = 5) 
    (point_x : f.x₀ = -1) 
    (point_y : f.y₀ = -20) : 
    f.a = -25/9 := by
  sorry


end quadratic_a_value_l699_69939


namespace mary_potatoes_l699_69909

def total_potatoes (initial : ℕ) (remaining_new : ℕ) : ℕ :=
  initial + remaining_new

theorem mary_potatoes : total_potatoes 8 3 = 11 := by
  sorry

end mary_potatoes_l699_69909


namespace divisor_problem_l699_69912

theorem divisor_problem (p n : ℕ) (d : Fin 8 → ℕ) : 
  p.Prime → 
  n > 0 →
  (∀ i : Fin 8, d i > 0) →
  (∀ i j : Fin 8, i < j → d i < d j) →
  d 0 = 1 →
  d 7 = p * n →
  (∀ x : ℕ, x ∣ (p * n) ↔ ∃ i : Fin 8, d i = x) →
  d (⟨17 * p - d 2, sorry⟩ : Fin 8) = (d 0 + d 1 + d 2) * (d 2 + d 3 + 13 * p) →
  n = 2021 := by
sorry

end divisor_problem_l699_69912


namespace carbonated_water_percentage_carbonated_water_percentage_proof_l699_69972

theorem carbonated_water_percentage : ℝ → Prop :=
  λ x =>
    let first_solution_carbonated := 0.80
    let mixture_carbonated := 0.65
    let first_solution_ratio := 0.40
    let second_solution_ratio := 0.60
    first_solution_carbonated * first_solution_ratio + x * second_solution_ratio = mixture_carbonated →
    x = 0.55

-- The proof is omitted
theorem carbonated_water_percentage_proof : carbonated_water_percentage 0.55 := by sorry

end carbonated_water_percentage_carbonated_water_percentage_proof_l699_69972


namespace smallest_angle_is_22_5_degrees_l699_69934

def smallest_positive_angle (y : ℝ) : Prop :=
  6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3/2 ∧
  y > 0 ∧
  ∀ z, z > 0 → 6 * Real.sin z * (Real.cos z)^3 - 6 * (Real.sin z)^3 * Real.cos z = 3/2 → y ≤ z

theorem smallest_angle_is_22_5_degrees :
  ∃ y, smallest_positive_angle y ∧ y = 22.5 * π / 180 :=
sorry

end smallest_angle_is_22_5_degrees_l699_69934


namespace polynomial_remainder_theorem_l699_69955

theorem polynomial_remainder_theorem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 - 3 * x^2 + b * x - 7
  (f 2 = -17) → (f (-1) = -11) → (a = 0 ∧ b = -1) := by
  sorry

end polynomial_remainder_theorem_l699_69955


namespace problem_statement_l699_69995

theorem problem_statement (x₁ x₂ x₃ a : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁)
  (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0)
  (h_eq1 : x₁ * x₂ - a * x₁ + a^2 = 0)
  (h_eq2 : x₂ * x₃ - a * x₂ + a^2 = 0) :
  (x₃ * x₁ - a * x₃ + a^2 = 0) ∧
  (x₁ * x₂ * x₃ + a^3 = 0) ∧
  (1 / (x₁ - x₂) + 1 / (x₂ - x₃) + 1 / (x₃ - x₁) = 1 / a) := by
  sorry

end problem_statement_l699_69995


namespace odd_indexed_sum_limit_l699_69933

/-- For an infinite geometric sequence {a_n} where a_1 = √3 and a_2 = 1,
    the limit of the sum of odd-indexed terms as n approaches infinity is (3√3)/2 -/
theorem odd_indexed_sum_limit (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = Real.sqrt 3 →                      -- first term condition
  a 2 = 1 →                                -- second term condition
  (∑' n : ℕ, a (2 * n + 1)) = 3 * Real.sqrt 3 / 2 := by
sorry

end odd_indexed_sum_limit_l699_69933


namespace quadratic_symmetry_l699_69999

/-- A quadratic function with axis of symmetry at x = 9 and p(6) = 2 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c (9 - x) = p a b c (9 + x)) →
  p a b c 6 = 2 →
  p a b c 12 = 2 := by
sorry

end quadratic_symmetry_l699_69999


namespace inequality_and_optimization_l699_69964

theorem inequality_and_optimization (a b m n : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n) 
  (hx : 0 < x ∧ x < 1/2) : 
  (m^2 / a + n^2 / b ≥ (m + n)^2 / (a + b)) ∧ 
  (∀ y ∈ Set.Ioo 0 (1/2), 2/x + 9/(1-2*x) ≤ 2/y + 9/(1-2*y)) ∧
  (2/x + 9/(1-2*x) = 25) ∧ (x = 1/5) := by
  sorry

end inequality_and_optimization_l699_69964


namespace min_students_in_both_clubs_l699_69991

theorem min_students_in_both_clubs 
  (total_students : ℕ) 
  (num_clubs : ℕ) 
  (min_percentage : ℚ) 
  (h1 : total_students = 33) 
  (h2 : num_clubs = 2) 
  (h3 : min_percentage = 7/10) : 
  ∃ (students_in_both : ℕ), 
    students_in_both ≥ 15 ∧ 
    ∀ (n1 n2 : ℕ), 
      n1 ≥ Int.ceil (total_students * min_percentage) → 
      n2 ≥ Int.ceil (total_students * min_percentage) → 
      n1 + n2 - students_in_both ≤ total_students → 
      students_in_both ≥ 15 :=
by sorry

end min_students_in_both_clubs_l699_69991


namespace sum_of_w_and_y_is_six_l699_69985

theorem sum_of_w_and_y_is_six (W X Y Z : ℕ) : 
  W ∈ ({1, 2, 3, 5} : Set ℕ) → 
  X ∈ ({1, 2, 3, 5} : Set ℕ) → 
  Y ∈ ({1, 2, 3, 5} : Set ℕ) → 
  Z ∈ ({1, 2, 3, 5} : Set ℕ) → 
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / X + (Y : ℚ) / Z = 3 →
  W + Y = 6 := by
sorry

end sum_of_w_and_y_is_six_l699_69985


namespace triangle_reflection_area_sum_l699_69975

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define reflection about a point
def reflect (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define area of a triangle
def area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_reflection_area_sum (t : Triangle) :
  let O := circumcenter t
  let A' := reflect t.A O
  let B' := reflect t.B O
  let C' := reflect t.C O
  area A' t.B t.C + area t.A B' t.C + area t.A t.B C' = area t.A t.B t.C := by
  sorry

end triangle_reflection_area_sum_l699_69975


namespace inequality_solution_set_l699_69929

-- Define the inequality
def inequality (x k : ℝ) : Prop := x^2 > (k+1)*x - k

-- Define the solution set
def solution_set (k : ℝ) : Set ℝ :=
  if k > 1 then {x : ℝ | x < 1 ∨ x > k}
  else if k = 1 then {x : ℝ | x ≠ 1}
  else {x : ℝ | x < k ∨ x > 1}

-- Theorem statement
theorem inequality_solution_set (k : ℝ) :
  {x : ℝ | inequality x k} = solution_set k :=
sorry

end inequality_solution_set_l699_69929


namespace scientific_notation_of_13976000_l699_69989

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_13976000 :
  toScientificNotation 13976000 = ScientificNotation.mk 1.3976 7 (by sorry) :=
sorry

end scientific_notation_of_13976000_l699_69989


namespace playground_children_count_l699_69947

theorem playground_children_count : 
  let boys : ℕ := 27
  let girls : ℕ := 35
  let total_children : ℕ := boys + girls
  total_children = 62 := by
  sorry

end playground_children_count_l699_69947


namespace average_cost_before_gratuity_l699_69941

/-- Given a group of 10 people with a total bill including 25% gratuity,
    calculate the average cost per person before gratuity. -/
theorem average_cost_before_gratuity
  (X : ℝ) -- Total bill including gratuity
  (h : X > 0) -- Assume the bill is positive
  : (X / 12.5 : ℝ) = (X / (1.25 * 10) : ℝ) := by
  sorry

#check average_cost_before_gratuity

end average_cost_before_gratuity_l699_69941


namespace valid_permutations_count_l699_69984

/-- Represents a permutation of 8 people around a circular table. -/
def CircularPermutation := Fin 8 → Fin 8

/-- Checks if two positions are adjacent on a circular table with 8 positions. -/
def is_adjacent (a b : Fin 8) : Prop :=
  (a - b) % 8 = 1 ∨ (b - a) % 8 = 1

/-- Checks if two positions are opposite on a circular table with 8 positions. -/
def is_opposite (a b : Fin 8) : Prop :=
  (a - b) % 8 = 4

/-- Checks if a permutation is valid according to the problem conditions. -/
def is_valid_permutation (p : CircularPermutation) : Prop :=
  ∀ i : Fin 8, 
    p i ≠ i ∧ 
    ¬is_adjacent i (p i) ∧ 
    ¬is_opposite i (p i)

/-- The main theorem stating that there are exactly 3 valid permutations. -/
theorem valid_permutations_count :
  ∃! (perms : Finset CircularPermutation),
    (∀ p ∈ perms, is_valid_permutation p) ∧
    perms.card = 3 :=
sorry

end valid_permutations_count_l699_69984


namespace normal_dist_prob_ge_six_l699_69978

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a random variable falls within one standard deviation of the mean -/
def prob_within_one_std (X : NormalDistribution) : ℝ := 0.6826

/-- The probability that a random variable is greater than or equal to a given value -/
noncomputable def prob_ge (X : NormalDistribution) (x : ℝ) : ℝ :=
  1 - (prob_within_one_std X) / 2

/-- Theorem: For a normal distribution N(5, 1), P(X ≥ 6) = 0.1587 -/
theorem normal_dist_prob_ge_six (X : NormalDistribution) 
  (h1 : X.μ = 5) (h2 : X.σ = 1) : 
  prob_ge X 6 = 0.1587 := by
  sorry


end normal_dist_prob_ge_six_l699_69978


namespace polynomial_remainder_theorem_l699_69976

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^5 + 2*x^3 + x + 3) % (x - 2) = 53 := by
sorry

end polynomial_remainder_theorem_l699_69976


namespace wall_volume_calculation_l699_69926

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  width : ℝ
  height : ℝ
  length : ℝ

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ := w.width * w.height * w.length

/-- Theorem stating the volume of the wall under given conditions -/
theorem wall_volume_calculation :
  ∃ (w : WallDimensions),
    w.width = 8 ∧
    w.height = 6 * w.width ∧
    w.length = 7 * w.height ∧
    wallVolume w = 128512 := by
  sorry


end wall_volume_calculation_l699_69926


namespace building_height_l699_69908

/-- Given a building and a pole casting shadows, calculates the height of the building. -/
theorem building_height (building_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  building_shadow = 63 →
  pole_height = 14 →
  pole_shadow = 18 →
  (building_shadow / pole_shadow) * pole_height = 49 := by
  sorry

#check building_height

end building_height_l699_69908


namespace sum_of_products_divisible_by_2011_l699_69911

def P (A : Finset ℕ) : ℕ := A.prod id

theorem sum_of_products_divisible_by_2011 : 
  ∃ (S : Finset (Finset ℕ)), 
    S.card = Nat.choose 2010 99 ∧ 
    (∀ A ∈ S, A.card = 99 ∧ A ⊆ Finset.range 2011) ∧
    2011 ∣ S.sum P :=
by sorry

end sum_of_products_divisible_by_2011_l699_69911


namespace trig_identities_and_expression_l699_69943

/-- Given an angle α whose terminal side passes through point (4, -3),
    prove the trigonometric identities and the value of a specific expression. -/
theorem trig_identities_and_expression (α : Real) 
  (h : ∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) : 
  Real.sin α = -3/5 ∧ 
  Real.cos α = 4/5 ∧ 
  Real.tan α = -3/4 ∧
  (Real.sin (Real.pi + α) + 2 * Real.sin (Real.pi/2 - α)) / (2 * Real.cos (Real.pi - α)) = -11/8 := by
  sorry

end trig_identities_and_expression_l699_69943


namespace shelby_total_stars_l699_69956

/-- The number of gold stars Shelby earned yesterday -/
def yesterdays_stars : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def todays_stars : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := yesterdays_stars + todays_stars

/-- Theorem: The total number of gold stars Shelby earned is 7 -/
theorem shelby_total_stars : total_stars = 7 := by
  sorry

end shelby_total_stars_l699_69956


namespace larger_integer_problem_l699_69944

theorem larger_integer_problem (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : y - x = 8) (h5 : x * y = 168) : y = 14 := by
  sorry

end larger_integer_problem_l699_69944


namespace complex_exp_angle_l699_69953

theorem complex_exp_angle (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → ∃ r θ : ℝ, z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end complex_exp_angle_l699_69953


namespace thirtieth_term_of_sequence_l699_69904

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h1 : a₁ = 3) (h2 : a₂ = 13) (h3 : a₃ = 23) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 293 := by
sorry

end thirtieth_term_of_sequence_l699_69904


namespace stock_comparison_l699_69998

/-- Represents the final value of a stock after two years of changes --/
def final_value (initial : ℚ) (change1 : ℚ) (change2 : ℚ) : ℚ :=
  initial * (1 + change1) * (1 + change2)

/-- The problem statement --/
theorem stock_comparison : 
  let A := final_value 150 0.1 (-0.05)
  let B := final_value 100 (-0.3) 0.1
  let C := final_value 50 0 0.08
  C < B ∧ B < A := by sorry

end stock_comparison_l699_69998


namespace train_length_l699_69961

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 6 → time = 2 → speed * time * (1000 / 3600) = 10 / 3 := by
  sorry

#check train_length

end train_length_l699_69961


namespace log_sum_sqrt_equality_l699_69990

theorem log_sum_sqrt_equality :
  Real.sqrt (Real.log 12 / Real.log 3 + Real.log 12 / Real.log 4) =
  Real.sqrt (Real.log 3 / Real.log 4) + Real.sqrt (Real.log 4 / Real.log 3) := by
  sorry

end log_sum_sqrt_equality_l699_69990


namespace complex_fraction_simplification_l699_69992

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 3*i) / (2 - i) = 1 - i := by sorry

end complex_fraction_simplification_l699_69992


namespace milk_cartons_sold_l699_69937

theorem milk_cartons_sold (regular : ℕ) (chocolate : ℕ) : 
  regular = 3 →
  chocolate = 7 * regular →
  regular + chocolate = 24 :=
by
  sorry

end milk_cartons_sold_l699_69937


namespace abc_values_l699_69949

theorem abc_values (a b c : ℝ) 
  (h1 : a / 2 = b / 3)
  (h2 : b / 3 = c / 4)
  (h3 : a / 2 ≠ 0)
  (h4 : 2 * a - b + c = 10) :
  a = 4 ∧ b = 6 ∧ c = 8 := by
sorry

end abc_values_l699_69949


namespace amy_cupcake_packages_l699_69920

def cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) : ℕ :=
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package

theorem amy_cupcake_packages :
  cupcake_packages 50 5 5 = 9 := by
  sorry

end amy_cupcake_packages_l699_69920


namespace weather_forecast_probability_meaning_l699_69917

/-- Represents the probability of an event -/
def Probability := ℝ

/-- Represents a weather forecast statement -/
structure WeatherForecast where
  statement : String
  probability : Probability

/-- Represents the meaning of a probability statement -/
inductive ProbabilityMeaning
  | Possibility
  | TimePercentage
  | AreaPercentage
  | PeopleOpinion

/-- The correct interpretation of a probability statement in a weather forecast -/
def correct_interpretation : ProbabilityMeaning := ProbabilityMeaning.Possibility

/-- 
  Theorem: The statement "The probability of rain tomorrow in this area is 80%" 
  in a weather forecast means "The possibility of rain tomorrow in this area is 80%"
-/
theorem weather_forecast_probability_meaning 
  (forecast : WeatherForecast) 
  (h : forecast.statement = "The probability of rain tomorrow in this area is 80%") :
  correct_interpretation = ProbabilityMeaning.Possibility := by sorry

end weather_forecast_probability_meaning_l699_69917


namespace bucket_6_5_full_l699_69977

/-- Represents the state of water in buckets -/
structure BucketState where
  bucket_2_5 : Real
  bucket_3 : Real
  bucket_5_6 : Real
  bucket_6_5 : Real

/-- Represents the capacity of each bucket -/
def bucket_capacities : BucketState :=
  { bucket_2_5 := 2.5
  , bucket_3 := 3
  , bucket_5_6 := 5.6
  , bucket_6_5 := 6.5 }

/-- Performs the water pouring operations as described in the problem -/
def pour_water (initial : BucketState) : BucketState :=
  sorry

/-- The theorem to be proved -/
theorem bucket_6_5_full (initial : BucketState) :
  let final := pour_water initial
  final.bucket_6_5 = bucket_capacities.bucket_6_5 ∧
  ∀ ε > 0, final.bucket_6_5 + ε > bucket_capacities.bucket_6_5 :=
sorry

end bucket_6_5_full_l699_69977


namespace adjacent_chair_subsets_l699_69906

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- The function that calculates the number of subsets with at least three adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  -- Subsets with exactly 3, 4, 5, or 6 adjacent chairs
  4 * n +
  -- Subsets with 7 or more chairs (always contain at least 3 adjacent)
  (Nat.choose n 7) + (Nat.choose n 8) + (Nat.choose n 9) +
  (Nat.choose n 10) + (Nat.choose n 11) + (Nat.choose n 12)

/-- The theorem stating that for 12 chairs, there are 1634 subsets with at least three adjacent chairs -/
theorem adjacent_chair_subsets :
  subsets_with_adjacent_chairs n = 1634 := by
  sorry

end adjacent_chair_subsets_l699_69906


namespace cookie_average_is_14_l699_69994

def cookie_counts : List Nat := [8, 10, 12, 15, 16, 17, 20]

def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem cookie_average_is_14 :
  average cookie_counts = 14 := by
  sorry

end cookie_average_is_14_l699_69994


namespace senior_ticket_cost_l699_69932

theorem senior_ticket_cost 
  (total_tickets : ℕ) 
  (regular_ticket_cost : ℕ) 
  (total_sales : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 65)
  (h2 : regular_ticket_cost = 15)
  (h3 : total_sales = 855)
  (h4 : senior_tickets = 24) :
  ∃ (senior_ticket_cost : ℕ),
    senior_ticket_cost * senior_tickets + 
    regular_ticket_cost * (total_tickets - senior_tickets) = 
    total_sales ∧ senior_ticket_cost = 10 :=
by sorry

end senior_ticket_cost_l699_69932


namespace hyperbola_condition_l699_69968

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m^2 + 1) - y^2 / (m^2 - 3) = 1) → 
  (m < -Real.sqrt 3 ∨ m > Real.sqrt 3) := by
sorry

end hyperbola_condition_l699_69968


namespace rectangle_area_l699_69900

/-- Given a rectangle with perimeter 14 cm and diagonal 5 cm, its area is 12 square centimeters. -/
theorem rectangle_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 14) 
  (h_diagonal : l^2 + w^2 = 5^2) : l * w = 12 := by
  sorry

end rectangle_area_l699_69900


namespace class_size_l699_69940

theorem class_size (initial_avg : ℝ) (wrong_score : ℝ) (correct_score : ℝ) (new_avg : ℝ) :
  initial_avg = 87.26 →
  wrong_score = 89 →
  correct_score = 98 →
  new_avg = 87.44 →
  ∃ n : ℕ, n = 50 ∧ n * new_avg = n * initial_avg + (correct_score - wrong_score) :=
by sorry

end class_size_l699_69940


namespace sum_of_squares_l699_69981

theorem sum_of_squares (x y z : ℝ) (h_positive : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end sum_of_squares_l699_69981


namespace circle_integers_l699_69993

/-- Given n integers equally spaced around a circle, if the diameter through 7 also goes through 23, then n = 32 -/
theorem circle_integers (n : ℕ) : n ≥ 23 → (∃ (k : ℕ), n = 2 * k ∧ (23 - 7) * 2 + 2 = n) → n = 32 := by
  sorry

end circle_integers_l699_69993


namespace sprocket_production_rate_l699_69935

theorem sprocket_production_rate : ∀ (a g : ℝ),
  -- Machine G produces 10% more sprockets per hour than Machine A
  g = 1.1 * a →
  -- Machine A takes 10 hours longer to produce 660 sprockets
  660 / a = 660 / g + 10 →
  -- The production rate of Machine A
  a = 6 := by
sorry

end sprocket_production_rate_l699_69935


namespace exponent_problem_l699_69974

theorem exponent_problem (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(m + n) = 6 ∧ a^(3*m - 2*n) = 8/9 := by
  sorry

end exponent_problem_l699_69974


namespace simplify_expression_1_simplify_expression_2_l699_69914

-- First expression
theorem simplify_expression_1 (a b : ℤ) :
  2*a - 6*b - 3*a + 9*b = -a + 3*b := by sorry

-- Second expression
theorem simplify_expression_2 (m n : ℤ) :
  2*(3*m^2 - m*n) - m*n + m^2 = 7*m^2 - 3*m*n := by sorry

end simplify_expression_1_simplify_expression_2_l699_69914


namespace inequality_solution_length_l699_69988

/-- Given an inequality c ≤ 3x + 5 ≤ d, where the length of the interval of solutions is 15, prove that d - c = 45 -/
theorem inequality_solution_length (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3*x + 5 ∧ 3*x + 5 ≤ d) → 
  ((d - 5) / 3 - (c - 5) / 3 = 15) →
  d - c = 45 := by
sorry

end inequality_solution_length_l699_69988


namespace complex_number_plus_modulus_l699_69918

theorem complex_number_plus_modulus (z : ℂ) : 
  z + Complex.abs z = 5 + Complex.I * Real.sqrt 3 → z = 11/5 + Complex.I * Real.sqrt 3 := by
  sorry

end complex_number_plus_modulus_l699_69918


namespace prob_excellent_given_pass_is_correct_l699_69969

/-- The number of total questions in the exam -/
def total_questions : ℕ := 20

/-- The number of questions selected for the exam -/
def selected_questions : ℕ := 6

/-- The number of questions the student can correctly answer -/
def correct_answers : ℕ := 10

/-- The minimum number of correct answers required to pass the exam -/
def pass_threshold : ℕ := 4

/-- The minimum number of correct answers required for an excellent grade -/
def excellent_threshold : ℕ := 5

/-- The probability of achieving an excellent grade given that the student has passed the exam -/
def prob_excellent_given_pass : ℚ := 13/58

/-- Theorem stating that the probability of achieving an excellent grade, 
    given that the student has passed the exam, is 13/58 -/
theorem prob_excellent_given_pass_is_correct :
  prob_excellent_given_pass = 13/58 := by
  sorry

end prob_excellent_given_pass_is_correct_l699_69969


namespace cube_side_length_l699_69945

theorem cube_side_length (s : ℝ) : s > 0 → (6 * s^2 = s^3) → s = 6 := by
  sorry

end cube_side_length_l699_69945


namespace polynomial_equality_l699_69938

theorem polynomial_equality (p : ℝ → ℝ) : 
  (∀ x, p x + (x^4 + 4*x^3 + 8*x) = (10*x^4 + 30*x^3 + 29*x^2 + 2*x + 5)) →
  (∀ x, p x = 9*x^4 + 26*x^3 + 29*x^2 - 6*x + 5) :=
by
  sorry

end polynomial_equality_l699_69938


namespace rectangular_solid_volume_l699_69902

theorem rectangular_solid_volume
  (a b c : ℕ+)
  (h1 : a * b - c * a - b * c = 1)
  (h2 : c * a = b * c + 1) :
  a * b * c = 6 :=
sorry

end rectangular_solid_volume_l699_69902


namespace polynomial_division_remainder_l699_69931

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 1 = (X^2 - 4*X + 7) * q + (8*X - 62) := by sorry

end polynomial_division_remainder_l699_69931


namespace regular_ngon_diagonal_difference_l699_69921

/-- The difference between the longest and shortest diagonals of a regular n-gon equals its side if and only if n = 9 -/
theorem regular_ngon_diagonal_difference (n : ℕ) : n ≥ 3 → (
  let R := (1 : ℝ)  -- Assume unit circumradius for simplicity
  let side_length := 2 * R * Real.sin (π / n)
  let shortest_diagonal := 2 * R * Real.sin (2 * π / n)
  let longest_diagonal := if n % 2 = 0 then 2 * R else 2 * R * Real.cos (π / (2 * n))
  longest_diagonal - shortest_diagonal = side_length
) ↔ n = 9 := by sorry

end regular_ngon_diagonal_difference_l699_69921


namespace items_in_bags_distribution_l699_69916

theorem items_in_bags_distribution (n : Nat) (k : Nat) : 
  n = 6 → k = 3 → (
    (1 : Nat) + -- All items in one bag
    n + -- n-1 items in one bag, 1 in another
    (n.choose 4) + -- 4 items in one bag, 2 in another
    (n.choose 4) + -- 4 items in one bag, 1 each in the other two
    (n.choose 3 / 2) + -- 3 items in each of two bags
    (n.choose 2 * (n - 2).choose 2 / 6) -- 2 items in each bag
  ) = 62 := by
  sorry

end items_in_bags_distribution_l699_69916
