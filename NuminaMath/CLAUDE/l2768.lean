import Mathlib

namespace NUMINAMATH_CALUDE_spying_arrangement_odd_l2768_276851

/-- A function representing the spying arrangement in a circular group -/
def spyingArrangement (n : ℕ) : ℕ → ℕ :=
  fun i => (i % n) + 1

/-- The theorem stating that the number of people in the spying arrangement must be odd -/
theorem spying_arrangement_odd (n : ℕ) (h : n > 0) :
  (∀ i : ℕ, i < n → spyingArrangement n (spyingArrangement n i) = (i + 2) % n + 1) →
  Odd n :=
sorry

end NUMINAMATH_CALUDE_spying_arrangement_odd_l2768_276851


namespace NUMINAMATH_CALUDE_product_of_sums_inequality_l2768_276849

theorem product_of_sums_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (x + 1) * (y + 1) ≥ 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_product_of_sums_inequality_l2768_276849


namespace NUMINAMATH_CALUDE_two_copresidents_probability_l2768_276815

def club_sizes : List Nat := [6, 9, 10]
def copresident_counts : List Nat := [2, 3, 2]

def probability_two_copresidents (sizes : List Nat) (copresidents : List Nat) : ℚ :=
  let probabilities := List.zipWith (fun n p =>
    (Nat.choose p 2 * Nat.choose (n - p) 2) / Nat.choose n 4
  ) sizes copresidents
  (1 / 3 : ℚ) * (probabilities.sum)

theorem two_copresidents_probability :
  probability_two_copresidents club_sizes copresident_counts = 11 / 42 := by
  sorry

end NUMINAMATH_CALUDE_two_copresidents_probability_l2768_276815


namespace NUMINAMATH_CALUDE_solve_for_n_l2768_276852

theorem solve_for_n (P s k m n : ℝ) (h : P = s / (1 + k + m) ^ n) :
  n = Real.log (s / P) / Real.log (1 + k + m) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_n_l2768_276852


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2768_276872

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 3 * a 5 = 64) :
  a 4 = 8 ∨ a 4 = -8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2768_276872


namespace NUMINAMATH_CALUDE_points_collinear_l2768_276889

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Check if four points are collinear -/
def are_collinear (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (t1 t2 t3 : ℝ), p2 = Point3D.mk (p1.x + t1 * (p4.x - p1.x)) (p1.y + t1 * (p4.y - p1.y)) (p1.z + t1 * (p4.z - p1.z)) ∧
                     p3 = Point3D.mk (p1.x + t2 * (p4.x - p1.x)) (p1.y + t2 * (p4.y - p1.y)) (p1.z + t2 * (p4.z - p1.z)) ∧
                     p4 = Point3D.mk (p1.x + t3 * (p4.x - p1.x)) (p1.y + t3 * (p4.y - p1.y)) (p1.z + t3 * (p4.z - p1.z))

/-- Main theorem -/
theorem points_collinear (pyramid : TriangularPyramid) 
  (M K P H E F Q T : Point3D)
  (h1 : (pyramid.A.x - M.x)^2 + (pyramid.A.y - M.y)^2 + (pyramid.A.z - M.z)^2 = 
        (M.x - K.x)^2 + (M.y - K.y)^2 + (M.z - K.z)^2)
  (h2 : (M.x - K.x)^2 + (M.y - K.y)^2 + (M.z - K.z)^2 = 
        (K.x - pyramid.D.x)^2 + (K.y - pyramid.D.y)^2 + (K.z - pyramid.D.z)^2)
  (h3 : (pyramid.B.x - P.x)^2 + (pyramid.B.y - P.y)^2 + (pyramid.B.z - P.z)^2 = 
        (P.x - H.x)^2 + (P.y - H.y)^2 + (P.z - H.z)^2)
  (h4 : (P.x - H.x)^2 + (P.y - H.y)^2 + (P.z - H.z)^2 = 
        (H.x - pyramid.C.x)^2 + (H.y - pyramid.C.y)^2 + (H.z - pyramid.C.z)^2)
  (h5 : (pyramid.A.x - E.x)^2 + (pyramid.A.y - E.y)^2 + (pyramid.A.z - E.z)^2 = 
        0.25 * ((pyramid.A.x - pyramid.B.x)^2 + (pyramid.A.y - pyramid.B.y)^2 + (pyramid.A.z - pyramid.B.z)^2))
  (h6 : (M.x - F.x)^2 + (M.y - F.y)^2 + (M.z - F.z)^2 = 
        0.25 * ((M.x - P.x)^2 + (M.y - P.y)^2 + (M.z - P.z)^2))
  (h7 : (K.x - Q.x)^2 + (K.y - Q.y)^2 + (K.z - Q.z)^2 = 
        0.25 * ((K.x - H.x)^2 + (K.y - H.y)^2 + (K.z - H.z)^2))
  (h8 : (pyramid.D.x - T.x)^2 + (pyramid.D.y - T.y)^2 + (pyramid.D.z - T.z)^2 = 
        0.25 * ((pyramid.D.x - pyramid.C.x)^2 + (pyramid.D.y - pyramid.C.y)^2 + (pyramid.D.z - pyramid.C.z)^2))
  : are_collinear E F Q T :=
sorry

end NUMINAMATH_CALUDE_points_collinear_l2768_276889


namespace NUMINAMATH_CALUDE_binary_operation_equality_l2768_276804

/-- Convert a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Convert a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Perform binary multiplication -/
def binary_mult (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

/-- Perform binary division -/
def binary_div (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a / binary_to_decimal b)

theorem binary_operation_equality : 
  let a := [true, true, false, false, true, false]  -- 110010₂
  let b := [true, true, false, false]               -- 1100₂
  let c := [true, false, false]                     -- 100₂
  let d := [true, false]                            -- 10₂
  let result := [true, false, false, true, false, false]  -- 100100₂
  binary_div (binary_div (binary_mult a b) c) d = result := by
  sorry

end NUMINAMATH_CALUDE_binary_operation_equality_l2768_276804


namespace NUMINAMATH_CALUDE_fraction_of_wall_painted_l2768_276835

/-- 
Given that a wall can be painted in 60 minutes, 
this theorem proves that the fraction of the wall 
painted in 15 minutes is 1/4.
-/
theorem fraction_of_wall_painted 
  (total_time : ℕ) 
  (partial_time : ℕ) 
  (h1 : total_time = 60) 
  (h2 : partial_time = 15) : 
  (partial_time : ℚ) / total_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_wall_painted_l2768_276835


namespace NUMINAMATH_CALUDE_ordinary_time_rate_l2768_276870

/-- Calculates the ordinary time rate given total hours, overtime hours, overtime rate, and total earnings -/
theorem ordinary_time_rate 
  (total_hours : ℕ) 
  (overtime_hours : ℕ) 
  (overtime_rate : ℚ) 
  (total_earnings : ℚ) 
  (h1 : total_hours = 50)
  (h2 : overtime_hours = 8)
  (h3 : overtime_rate = 9/10)
  (h4 : total_earnings = 1620/50)
  (h5 : overtime_hours ≤ total_hours) :
  let ordinary_hours := total_hours - overtime_hours
  let ordinary_rate := (total_earnings - overtime_rate * overtime_hours) / ordinary_hours
  ordinary_rate = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ordinary_time_rate_l2768_276870


namespace NUMINAMATH_CALUDE_f_of_one_eq_two_l2768_276818

def f (x : ℝ) := x^2 + |x - 2|

theorem f_of_one_eq_two : f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_of_one_eq_two_l2768_276818


namespace NUMINAMATH_CALUDE_red_balls_count_l2768_276876

/-- Given a set of balls where the ratio of red balls to white balls is 4:5,
    and there are 20 white balls, prove that the number of red balls is 16. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (white : ℕ) 
    (h1 : total = red + white)
    (h2 : red * 5 = white * 4)
    (h3 : white = 20) : red = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2768_276876


namespace NUMINAMATH_CALUDE_muffin_fundraiser_l2768_276861

/-- Proves the number of muffin cases needed to raise $120 --/
theorem muffin_fundraiser (muffins_per_pack : ℕ) (packs_per_case : ℕ) 
  (price_per_muffin : ℚ) (fundraising_goal : ℚ) :
  muffins_per_pack = 4 →
  packs_per_case = 3 →
  price_per_muffin = 2 →
  fundraising_goal = 120 →
  (fundraising_goal / (muffins_per_pack * packs_per_case * price_per_muffin) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_muffin_fundraiser_l2768_276861


namespace NUMINAMATH_CALUDE_division_remainder_l2768_276821

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 181 → 
  divisor = 20 → 
  quotient = 9 → 
  remainder = dividend - (divisor * quotient) → 
  remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2768_276821


namespace NUMINAMATH_CALUDE_gp_common_ratio_l2768_276811

theorem gp_common_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28 →
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_gp_common_ratio_l2768_276811


namespace NUMINAMATH_CALUDE_new_cube_edge_length_l2768_276802

-- Define the edge lengths of the original cubes
def edge1 : ℝ := 6
def edge2 : ℝ := 8
def edge3 : ℝ := 10

-- Define the volume of a cube given its edge length
def cubeVolume (edge : ℝ) : ℝ := edge ^ 3

-- Define the total volume of the three original cubes
def totalVolume : ℝ := cubeVolume edge1 + cubeVolume edge2 + cubeVolume edge3

-- Define the edge length of the new cube
def newEdge : ℝ := totalVolume ^ (1/3)

-- Theorem statement
theorem new_cube_edge_length : newEdge = 12 := by
  sorry

end NUMINAMATH_CALUDE_new_cube_edge_length_l2768_276802


namespace NUMINAMATH_CALUDE_original_fraction_l2768_276871

theorem original_fraction (N D : ℚ) : 
  (N * (1 + 30/100)) / (D * (1 - 15/100)) = 25/21 →
  N / D = 425/546 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l2768_276871


namespace NUMINAMATH_CALUDE_g_100_zeros_l2768_276842

-- Define g₀(x)
def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

-- Define gₙ(x) recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_100_zeros :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ g 100 x = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_100_zeros_l2768_276842


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2768_276820

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop :=
  z * i = ((i + 1) / (i - 1)) ^ 2016

-- Theorem statement
theorem complex_equation_solution :
  ∃ (z : ℂ), equation z ∧ z = -i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2768_276820


namespace NUMINAMATH_CALUDE_unique_solution_l2768_276883

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2768_276883


namespace NUMINAMATH_CALUDE_unique_nonzero_elements_in_rows_and_columns_l2768_276803

open Matrix

theorem unique_nonzero_elements_in_rows_and_columns
  (n : ℕ)
  (A : Matrix (Fin n) (Fin n) ℝ)
  (h_nonneg : ∀ i j, 0 ≤ A i j)
  (h_nonsingular : IsUnit (det A))
  (h_inv_nonneg : ∀ i j, 0 ≤ A⁻¹ i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_nonzero_elements_in_rows_and_columns_l2768_276803


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l2768_276834

/-- A line tangent to the unit circle with intercept sum √3 forms a triangle with area 3/2 --/
theorem tangent_line_triangle_area :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y = 1) →  -- Line is tangent to unit circle
  a + b = Real.sqrt 3 →                           -- Sum of intercepts is √3
  (1/2) * |a*b| = 3/2 :=                          -- Area of triangle is 3/2
by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l2768_276834


namespace NUMINAMATH_CALUDE_regular_21gon_symmetry_sum_l2768_276848

theorem regular_21gon_symmetry_sum : 
  let n : ℕ := 21
  let L' : ℕ := n  -- number of lines of symmetry
  let R' : ℚ := 360 / n  -- smallest positive angle of rotational symmetry in degrees
  L' + R' = 38.142857 := by sorry

end NUMINAMATH_CALUDE_regular_21gon_symmetry_sum_l2768_276848


namespace NUMINAMATH_CALUDE_danny_initial_caps_l2768_276874

/-- The number of bottle caps Danny had initially -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Danny threw away -/
def thrown_away : ℕ := 60

/-- The number of new bottle caps Danny found -/
def found : ℕ := 58

/-- The number of bottle caps Danny traded away -/
def traded_away : ℕ := 15

/-- The number of bottle caps Danny received in trade -/
def received : ℕ := 25

/-- The number of bottle caps Danny has now -/
def final_caps : ℕ := 67

/-- Theorem stating that Danny initially had 59 bottle caps -/
theorem danny_initial_caps : 
  initial_caps = 59 ∧
  final_caps = initial_caps - thrown_away + found - traded_away + received :=
sorry

end NUMINAMATH_CALUDE_danny_initial_caps_l2768_276874


namespace NUMINAMATH_CALUDE_schoolClubProfit_l2768_276800

/-- Represents the candy bar sale scenario for a school club -/
structure CandyBarSale where
  totalBars : ℕ
  purchaseRate : ℚ
  regularSellRate : ℚ
  bulkSellRate : ℚ

/-- Calculates the profit for the candy bar sale -/
def calculateProfit (sale : CandyBarSale) : ℚ :=
  let costPerBar := sale.purchaseRate / 4
  let totalCost := costPerBar * sale.totalBars
  let revenuePerBar := sale.regularSellRate / 3
  let totalRevenue := revenuePerBar * sale.totalBars
  totalRevenue - totalCost

/-- The given candy bar sale scenario -/
def schoolClubSale : CandyBarSale :=
  { totalBars := 1200
  , purchaseRate := 3
  , regularSellRate := 2
  , bulkSellRate := 3/5 }

/-- Theorem stating that the profit for the school club is -100 dollars -/
theorem schoolClubProfit :
  calculateProfit schoolClubSale = -100 := by
  sorry


end NUMINAMATH_CALUDE_schoolClubProfit_l2768_276800


namespace NUMINAMATH_CALUDE_ninth_grade_class_problem_l2768_276888

theorem ninth_grade_class_problem (total : ℕ) (science : ℕ) (arts : ℕ) 
  (h_total : total = 120)
  (h_science : science = 85)
  (h_arts : arts = 65)
  (h_covers_all : total ≤ science + arts) :
  science - (science + arts - total) = 55 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_class_problem_l2768_276888


namespace NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l2768_276897

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 + 9*y^2 - 6*x + 18*y = k

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b h₁ h₂ : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), curve_equation x y k ↔ (x - h₁)^2 / a^2 + (y - h₂)^2 / b^2 = 1

/-- Theorem stating the condition for the curve to be a non-degenerate ellipse -/
theorem non_degenerate_ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -9 := by sorry

end NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l2768_276897


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2768_276896

theorem quadratic_function_properties (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2768_276896


namespace NUMINAMATH_CALUDE_rational_sqrt2_distance_l2768_276806

theorem rational_sqrt2_distance (a b : ℤ) (h₁ : b ≠ 0) (h₂ : 0 < a/b) (h₃ : a/b < 1) :
  |a/b - 1/Real.sqrt 2| > 1/(4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt2_distance_l2768_276806


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_function_l2768_276845

theorem min_value_of_quadratic_function :
  ∃ (min : ℝ), min = -1 ∧ ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x + 8 * y + 15 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_function_l2768_276845


namespace NUMINAMATH_CALUDE_find_t_value_l2768_276812

theorem find_t_value (s t : ℚ) 
  (eq1 : 12 * s + 7 * t = 154)
  (eq2 : s = 2 * t - 3) : 
  t = 190 / 31 := by
sorry

end NUMINAMATH_CALUDE_find_t_value_l2768_276812


namespace NUMINAMATH_CALUDE_max_values_f_and_g_l2768_276873

noncomputable def f (θ : ℝ) := (1 + Real.cos θ) * (1 + Real.sin θ)
noncomputable def g (θ : ℝ) := (1/2 + Real.cos θ) * (Real.sqrt 3/2 + Real.sin θ)

theorem max_values_f_and_g :
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) ∧ f θ = (3 + 2 * Real.sqrt 2)/2) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) → f θ ≤ (3 + 2 * Real.sqrt 2)/2) ∧
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) ∧ g θ = Real.sqrt 3/4 + 3/2 * Real.sin (5*Real.pi/9)) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) → g θ ≤ Real.sqrt 3/4 + 3/2 * Real.sin (5*Real.pi/9)) :=
by sorry

end NUMINAMATH_CALUDE_max_values_f_and_g_l2768_276873


namespace NUMINAMATH_CALUDE_litter_bag_weight_l2768_276866

theorem litter_bag_weight (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (total_weight : ℕ) :
  gina_bags = 2 →
  neighborhood_multiplier = 82 →
  total_weight = 664 →
  ∃ (bag_weight : ℕ), 
    bag_weight = 4 ∧ 
    (gina_bags + neighborhood_multiplier * gina_bags) * bag_weight = total_weight :=
by sorry

end NUMINAMATH_CALUDE_litter_bag_weight_l2768_276866


namespace NUMINAMATH_CALUDE_jackson_earned_thirty_dollars_l2768_276809

-- Define the rate of pay per hour
def pay_rate : ℝ := 5

-- Define the time spent on each chore
def vacuuming_time : ℝ := 2
def dish_washing_time : ℝ := 0.5
def bathroom_cleaning_time : ℝ := 3 * dish_washing_time

-- Define the number of times vacuuming is done
def vacuuming_repetitions : ℕ := 2

-- Calculate total chore time
def total_chore_time : ℝ :=
  vacuuming_time * vacuuming_repetitions + dish_washing_time + bathroom_cleaning_time

-- Calculate earned money
def earned_money : ℝ := total_chore_time * pay_rate

-- Theorem statement
theorem jackson_earned_thirty_dollars :
  earned_money = 30 :=
by sorry

end NUMINAMATH_CALUDE_jackson_earned_thirty_dollars_l2768_276809


namespace NUMINAMATH_CALUDE_james_total_earnings_l2768_276891

def january_earnings : ℕ := 4000

def february_earnings : ℕ := 2 * january_earnings

def march_earnings : ℕ := february_earnings - 2000

def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings : total_earnings = 18000 := by
  sorry

end NUMINAMATH_CALUDE_james_total_earnings_l2768_276891


namespace NUMINAMATH_CALUDE_age_difference_l2768_276833

/-- Proves that Sachin is 8 years younger than Rahul given their age ratio and Sachin's age -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 28 →
  (sachin_age : ℚ) / rahul_age = 7 / 9 →
  rahul_age - sachin_age = 8 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2768_276833


namespace NUMINAMATH_CALUDE_sum_of_terms_l2768_276801

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_terms (a : ℕ → ℕ) : 
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 31 →
  a 4 + a 5 + a 7 = 93 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_l2768_276801


namespace NUMINAMATH_CALUDE_exponential_curve_logarithm_relation_l2768_276828

/-- Proves the relationship between u, b, x, and c for an exponential curve -/
theorem exponential_curve_logarithm_relation 
  (a b x : ℝ) 
  (y : ℝ := a * Real.exp (b * x)) 
  (u : ℝ := Real.log y) 
  (c : ℝ := Real.log a) : 
  u = b * x + c := by
  sorry

end NUMINAMATH_CALUDE_exponential_curve_logarithm_relation_l2768_276828


namespace NUMINAMATH_CALUDE_percentage_problem_l2768_276853

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 2 * x) : x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2768_276853


namespace NUMINAMATH_CALUDE_smallest_number_with_gcd_six_l2768_276875

theorem smallest_number_with_gcd_six : ∃ (n : ℕ), 
  (70 ≤ n ∧ n ≤ 90) ∧ 
  Nat.gcd n 24 = 6 ∧ 
  (∀ m, (70 ≤ m ∧ m < n) → Nat.gcd m 24 ≠ 6) ∧
  n = 78 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_gcd_six_l2768_276875


namespace NUMINAMATH_CALUDE_female_rainbow_trout_count_l2768_276816

theorem female_rainbow_trout_count :
  -- Total speckled trout
  ∀ (total_speckled : ℕ),
  -- Male and female speckled trout
  ∀ (male_speckled female_speckled : ℕ),
  -- Male rainbow trout
  ∀ (male_rainbow : ℕ),
  -- Total trout
  ∀ (total_trout : ℕ),
  -- Conditions
  total_speckled = 645 →
  male_speckled = 2 * female_speckled + 45 →
  4 * male_rainbow = 3 * female_speckled →
  20 * male_rainbow = 3 * total_trout →
  -- Conclusion
  total_trout - total_speckled - male_rainbow = 205 :=
by
  sorry

end NUMINAMATH_CALUDE_female_rainbow_trout_count_l2768_276816


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l2768_276893

theorem power_three_mod_eleven : 3^87 + 5 ≡ 3 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l2768_276893


namespace NUMINAMATH_CALUDE_min_weighings_to_find_fake_pearl_l2768_276892

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal : WeighResult
  | Left : WeighResult
  | Right : WeighResult

/-- Represents a strategy for finding the fake pearl -/
def Strategy := List WeighResult → Nat

/-- The number of pearls -/
def numPearls : Nat := 9

/-- The minimum number of weighings needed to find the fake pearl -/
def minWeighings : Nat := 2

/-- A theorem stating that the minimum number of weighings to find the fake pearl is 2 -/
theorem min_weighings_to_find_fake_pearl :
  ∃ (s : Strategy), ∀ (outcomes : List WeighResult),
    outcomes.length ≤ minWeighings →
    s outcomes < numPearls ∧
    (∀ (t : Strategy),
      (∀ (outcomes' : List WeighResult),
        outcomes'.length < outcomes.length →
        t outcomes' = numPearls) →
      s outcomes ≤ t outcomes) :=
sorry

end NUMINAMATH_CALUDE_min_weighings_to_find_fake_pearl_l2768_276892


namespace NUMINAMATH_CALUDE_cookies_remaining_l2768_276824

/-- Represents the baked goods scenario --/
structure BakedGoods where
  cookies : ℕ
  brownies : ℕ
  cookie_price : ℚ
  brownie_price : ℚ

/-- Calculates the total value of baked goods --/
def total_value (bg : BakedGoods) : ℚ :=
  bg.cookies * bg.cookie_price + bg.brownies * bg.brownie_price

/-- Theorem stating the number of cookies remaining --/
theorem cookies_remaining (bg : BakedGoods) 
  (h1 : bg.brownies = 32)
  (h2 : bg.cookie_price = 1)
  (h3 : bg.brownie_price = 3/2)
  (h4 : total_value bg = 99) :
  bg.cookies = 51 := by
sorry


end NUMINAMATH_CALUDE_cookies_remaining_l2768_276824


namespace NUMINAMATH_CALUDE_tangent_line_at_two_range_of_m_for_three_roots_l2768_276886

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_two :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (x = 2 → A * x + B * y + C = 0)) ∧
  A = 12 ∧ B = -1 ∧ C = -17 := by sorry

-- Theorem for the range of m
theorem range_of_m_for_three_roots :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x + m = 0 ∧ f y + m = 0 ∧ f z + m = 0) ↔ 
  -3 < m ∧ m < -2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_two_range_of_m_for_three_roots_l2768_276886


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_zero_l2768_276854

def f (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem sum_of_max_min_is_zero :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_zero_l2768_276854


namespace NUMINAMATH_CALUDE_multiply_333_by_111_l2768_276877

theorem multiply_333_by_111 : 333 * 111 = 36963 := by
  sorry

end NUMINAMATH_CALUDE_multiply_333_by_111_l2768_276877


namespace NUMINAMATH_CALUDE_book_distribution_ways_l2768_276819

/-- The number of ways to distribute books to students -/
def distribute_books (num_book_types : ℕ) (num_students : ℕ) (min_copies : ℕ) : ℕ :=
  num_book_types ^ num_students

/-- Theorem: There are 125 ways to distribute 3 books to 3 students from 5 types of books -/
theorem book_distribution_ways :
  distribute_books 5 3 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l2768_276819


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l2768_276843

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x + 2 else 2 * x + a

-- State the theorem
theorem continuous_piecewise_function (a : ℝ) :
  Continuous (f a) ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l2768_276843


namespace NUMINAMATH_CALUDE_total_shells_is_83_l2768_276827

/-- The total number of shells in the combined collection of five friends -/
def total_shells (initial_shells : ℕ) 
  (ed_limpet ed_oyster ed_conch ed_scallop : ℕ)
  (jacob_extra : ℕ)
  (marissa_limpet marissa_oyster marissa_conch marissa_scallop : ℕ)
  (priya_clam priya_mussel priya_conch priya_oyster : ℕ)
  (carlos_shells : ℕ) : ℕ :=
  initial_shells + 
  (ed_limpet + ed_oyster + ed_conch + ed_scallop) + 
  (ed_limpet + ed_oyster + ed_conch + ed_scallop + jacob_extra) +
  (marissa_limpet + marissa_oyster + marissa_conch + marissa_scallop) +
  (priya_clam + priya_mussel + priya_conch + priya_oyster) +
  carlos_shells

/-- The theorem stating that the total number of shells is 83 -/
theorem total_shells_is_83 : 
  total_shells 2 7 2 4 3 2 5 6 3 1 8 4 3 2 15 = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_83_l2768_276827


namespace NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l2768_276867

theorem max_right_triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a = 12 ∧ b = 5 →
  (∀ (x y : ℝ),
    x ≤ a ∧ y ≤ b →
    x * y / 2 ≤ 30) ∧
  ∃ (x y : ℝ),
    x ≤ a ∧ y ≤ b ∧
    x * y / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_max_right_triangle_area_in_rectangle_l2768_276867


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2768_276856

/-- The sum of terms in an arithmetic sequence with first term 2, common difference 12, and last term 182 is 1472 -/
theorem arithmetic_sequence_sum : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 12  -- Common difference
  let aₙ : ℕ := 182  -- Last term
  let n : ℕ := (aₙ - a₁) / d + 1  -- Number of terms
  (n : ℝ) * (a₁ + aₙ) / 2 = 1472 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2768_276856


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2768_276858

theorem batsman_average_increase 
  (innings : Nat) 
  (last_score : Nat) 
  (final_average : Nat) 
  (h1 : innings = 12) 
  (h2 : last_score = 75) 
  (h3 : final_average = 64) : 
  (final_average : ℚ) - (((innings : ℚ) * final_average - last_score) / (innings - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l2768_276858


namespace NUMINAMATH_CALUDE_simplify_expression_l2768_276813

theorem simplify_expression : (2^8 + 7^3) * (2^2 - (-2)^3)^5 = 149062368 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2768_276813


namespace NUMINAMATH_CALUDE_smallest_multiple_l2768_276887

theorem smallest_multiple (x : ℕ+) : (∀ y : ℕ+, 450 * y.val % 625 = 0 → x ≤ y) ∧ 450 * x.val % 625 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2768_276887


namespace NUMINAMATH_CALUDE_dans_initial_cards_l2768_276881

/-- The number of baseball cards Dan had initially -/
def initial_cards : ℕ := sorry

/-- The number of torn cards -/
def torn_cards : ℕ := 8

/-- The number of cards Sam bought -/
def cards_sold : ℕ := 15

/-- The number of cards Dan has after selling to Sam -/
def remaining_cards : ℕ := 82

theorem dans_initial_cards : initial_cards = 105 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_cards_l2768_276881


namespace NUMINAMATH_CALUDE_quiz_bowl_points_per_answer_l2768_276850

/-- Represents the quiz bowl game structure and James' performance --/
structure QuizBowl where
  total_rounds : Nat
  questions_per_round : Nat
  bonus_points : Nat
  james_total_points : Nat
  james_missed_questions : Nat

/-- Calculates the points per correct answer in the quiz bowl --/
def points_per_correct_answer (qb : QuizBowl) : Nat :=
  let total_questions := qb.total_rounds * qb.questions_per_round
  let james_correct_answers := total_questions - qb.james_missed_questions
  let perfect_rounds := (james_correct_answers / qb.questions_per_round)
  let bonus_total := perfect_rounds * qb.bonus_points
  let points_from_answers := qb.james_total_points - bonus_total
  points_from_answers / james_correct_answers

/-- Theorem stating that given the specific conditions, the points per correct answer is 2 --/
theorem quiz_bowl_points_per_answer :
  let qb : QuizBowl := {
    total_rounds := 5,
    questions_per_round := 5,
    bonus_points := 4,
    james_total_points := 66,
    james_missed_questions := 1
  }
  points_per_correct_answer qb = 2 := by
  sorry

end NUMINAMATH_CALUDE_quiz_bowl_points_per_answer_l2768_276850


namespace NUMINAMATH_CALUDE_equation_solution_l2768_276878

theorem equation_solution : 
  let y : ℝ := -33/2
  ∀ x : ℝ, (8*x^2 + 78*x + 5) / (2*x + 19) = 4*x + 2 → x = y := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2768_276878


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_1812_l2768_276863

/-- Calculates the cost of white washing a room with given dimensions and openings. -/
def whitewashingCost (length width height : ℝ) (doorHeight doorWidth : ℝ) 
  (windowHeight windowWidth : ℝ) (numWindows : ℕ) (ratePerSqFt : ℝ) : ℝ :=
  let wallArea := 2 * (length + width) * height
  let doorArea := doorHeight * doorWidth
  let windowArea := windowHeight * windowWidth * (numWindows : ℝ)
  let adjustedArea := wallArea - doorArea - windowArea
  adjustedArea * ratePerSqFt

/-- The cost of white washing the room is Rs. 1812. -/
theorem whitewashing_cost_is_1812 :
  whitewashingCost 25 15 12 6 3 4 3 3 2 = 1812 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_1812_l2768_276863


namespace NUMINAMATH_CALUDE_xyz_value_l2768_276837

theorem xyz_value (x y z s : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 12)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 0)
  (h3 : x + y + z = s) :
  x * y * z = -8 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2768_276837


namespace NUMINAMATH_CALUDE_intersection_perimeter_constant_l2768_276894

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A plane parallel to two opposite edges of a regular tetrahedron -/
structure ParallelPlane (t : RegularTetrahedron) where
  -- We don't need to define the plane explicitly, just its existence

/-- The figure obtained from intersecting a regular tetrahedron with a parallel plane -/
def IntersectionFigure (t : RegularTetrahedron) (p : ParallelPlane t) : Type :=
  -- We don't need to define the figure explicitly, just its existence
  Unit

/-- The perimeter of an intersection figure -/
noncomputable def perimeter (t : RegularTetrahedron) (p : ParallelPlane t) (f : IntersectionFigure t p) : ℝ :=
  2 * t.edge_length

/-- Theorem: The perimeter of any intersection figure is equal to 2a -/
theorem intersection_perimeter_constant (t : RegularTetrahedron) 
  (p : ParallelPlane t) (f : IntersectionFigure t p) :
  perimeter t p f = 2 * t.edge_length :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_perimeter_constant_l2768_276894


namespace NUMINAMATH_CALUDE_coach_team_division_l2768_276831

/-- Given a total number of athletes and a maximum team size, 
    calculate the minimum number of teams needed. -/
def min_teams (total_athletes : ℕ) (max_team_size : ℕ) : ℕ :=
  ((total_athletes + max_team_size - 1) / max_team_size : ℕ)

theorem coach_team_division (total_athletes max_team_size : ℕ) 
  (h1 : total_athletes = 30) (h2 : max_team_size = 12) :
  min_teams total_athletes max_team_size = 3 := by
  sorry

#eval min_teams 30 12

end NUMINAMATH_CALUDE_coach_team_division_l2768_276831


namespace NUMINAMATH_CALUDE_min_value_expression_l2768_276808

theorem min_value_expression (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 ≥ 4 * (4:ℝ)^(1/5) ∧
  (3 * Real.sqrt x + 4 / x^2 = 4 * (4:ℝ)^(1/5) ↔ x = (4:ℝ)^(2/5)) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2768_276808


namespace NUMINAMATH_CALUDE_sin_180_degrees_is_zero_l2768_276817

/-- The sine of 180 degrees is 0 -/
theorem sin_180_degrees_is_zero : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_is_zero_l2768_276817


namespace NUMINAMATH_CALUDE_camp_men_count_l2768_276814

/-- The number of days the food lasts initially -/
def initial_days : ℕ := 50

/-- The number of days the food lasts after more men join -/
def final_days : ℕ := 25

/-- The number of additional men who join -/
def additional_men : ℕ := 10

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

theorem camp_men_count :
  ∀ (food : ℕ),
  food = initial_men * initial_days ∧
  food = (initial_men + additional_men) * final_days →
  initial_men = 10 := by
sorry

end NUMINAMATH_CALUDE_camp_men_count_l2768_276814


namespace NUMINAMATH_CALUDE_intersection_implies_a_range_l2768_276882

/-- Given two functions f and g, where f(x) = ax and g(x) = ln x, 
    if their graphs intersect at two different points in (0, +∞),
    then 0 < a < 1/e. -/
theorem intersection_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ 
   a * x₁ = Real.log x₁ ∧ a * x₂ = Real.log x₂) →
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_range_l2768_276882


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_lines_parallel_l2768_276884

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- Axioms for different objects
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem 1: Transitivity of parallel planes
theorem parallel_transitive :
  parallel α β → parallel β γ → parallel α γ :=
sorry

-- Theorem 2: Lines perpendicular to the same plane are parallel
theorem perpendicular_lines_parallel :
  perpendicular m α → perpendicular n α → lineParallel m n :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_lines_parallel_l2768_276884


namespace NUMINAMATH_CALUDE_smallest_n_congruence_eight_satisfies_congruence_eight_is_smallest_smallest_positive_integer_congruence_l2768_276810

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 5678 [ZMOD 11]) → n ≥ 8 :=
by sorry

theorem eight_satisfies_congruence : 19 * 8 ≡ 5678 [ZMOD 11] :=
by sorry

theorem eight_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 8 → ¬(19 * m ≡ 5678 [ZMOD 11]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 19 * n ≡ 5678 [ZMOD 11] ∧ 
  ∀ m : ℕ, m > 0 ∧ 19 * m ≡ 5678 [ZMOD 11] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_eight_satisfies_congruence_eight_is_smallest_smallest_positive_integer_congruence_l2768_276810


namespace NUMINAMATH_CALUDE_calculate_brads_speed_l2768_276855

/-- Given two people walking towards each other, calculate the speed of one person given the other's speed and distance traveled. -/
theorem calculate_brads_speed (maxwell_speed brad_speed : ℝ) (total_distance maxwell_distance : ℝ) : 
  maxwell_speed = 2 →
  total_distance = 36 →
  maxwell_distance = 12 →
  2 * maxwell_distance = total_distance →
  brad_speed = 4 := by sorry

end NUMINAMATH_CALUDE_calculate_brads_speed_l2768_276855


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2768_276822

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 30, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 360 := by
  sorry

#eval (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions)

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2768_276822


namespace NUMINAMATH_CALUDE_double_markup_percentage_l2768_276840

theorem double_markup_percentage (initial_price : ℝ) (markup_percentage : ℝ) : 
  markup_percentage = 40 →
  let first_markup := initial_price * (1 + markup_percentage / 100)
  let second_markup := first_markup * (1 + markup_percentage / 100)
  (second_markup - initial_price) / initial_price * 100 = 96 := by
  sorry

end NUMINAMATH_CALUDE_double_markup_percentage_l2768_276840


namespace NUMINAMATH_CALUDE_mean_score_proof_l2768_276830

theorem mean_score_proof (first_class_mean second_class_mean : ℝ)
                         (total_students : ℕ)
                         (class_ratio : ℚ) :
  first_class_mean = 90 →
  second_class_mean = 75 →
  total_students = 66 →
  class_ratio = 5 / 6 →
  ∃ (first_class_students second_class_students : ℕ),
    first_class_students + second_class_students = total_students ∧
    (first_class_students : ℚ) / (second_class_students : ℚ) = class_ratio ∧
    (first_class_mean * (first_class_students : ℝ) + 
     second_class_mean * (second_class_students : ℝ)) / (total_students : ℝ) = 82 :=
by sorry

end NUMINAMATH_CALUDE_mean_score_proof_l2768_276830


namespace NUMINAMATH_CALUDE_no_solution_exists_l2768_276838

theorem no_solution_exists (a b : ℝ) : a^2 + 3*b^2 + 2 > 3*a*b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2768_276838


namespace NUMINAMATH_CALUDE_matrix_identities_l2768_276807

variable {n : ℕ} (hn : n ≥ 2)
variable (k : ℝ)
variable (A B C D : Matrix (Fin n) (Fin n) ℂ)

theorem matrix_identities 
  (h1 : A * C + k • (B * D) = 1)
  (h2 : A * D = B * C) :
  C * A + k • (D * B) = 1 ∧ D * A = C * B := by
sorry

end NUMINAMATH_CALUDE_matrix_identities_l2768_276807


namespace NUMINAMATH_CALUDE_quadratic_equation_rewrite_l2768_276890

theorem quadratic_equation_rewrite :
  ∃ (a b c : ℝ), a = 2 ∧ b = -4 ∧ c = 7 ∧
  ∀ x, 2 * x^2 + 7 = 4 * x ↔ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rewrite_l2768_276890


namespace NUMINAMATH_CALUDE_base_score_per_round_l2768_276862

theorem base_score_per_round 
  (total_rounds : ℕ) 
  (total_points : ℕ) 
  (bonus_points : ℕ) 
  (penalty_points : ℕ) 
  (h1 : total_rounds = 5)
  (h2 : total_points = 370)
  (h3 : bonus_points = 50)
  (h4 : penalty_points = 30) :
  (total_points - bonus_points + penalty_points) / total_rounds = 70 := by
sorry

end NUMINAMATH_CALUDE_base_score_per_round_l2768_276862


namespace NUMINAMATH_CALUDE_circles_intersection_range_l2768_276880

/-- Two circles C₁ and C₂ defined by their equations -/
def C₁ (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 4 = 0
def C₂ (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0

/-- The condition for two circles to intersect -/
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, C₁ m x y ∧ C₂ m x y

/-- The theorem stating the range of m for which the circles intersect -/
theorem circles_intersection_range :
  ∀ m : ℝ, circles_intersect m ↔ (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_range_l2768_276880


namespace NUMINAMATH_CALUDE_truck_journey_distance_l2768_276895

/-- A problem about a semi truck's journey on paved and dirt roads. -/
theorem truck_journey_distance :
  let time_paved : ℝ := 2 -- Time spent on paved road in hours
  let time_dirt : ℝ := 3 -- Time spent on dirt road in hours
  let speed_dirt : ℝ := 32 -- Speed on dirt road in mph
  let speed_paved : ℝ := speed_dirt + 20 -- Speed on paved road in mph
  let distance_dirt : ℝ := speed_dirt * time_dirt -- Distance on dirt road
  let distance_paved : ℝ := speed_paved * time_paved -- Distance on paved road
  let total_distance : ℝ := distance_dirt + distance_paved -- Total distance of the trip
  total_distance = 200 := by sorry

end NUMINAMATH_CALUDE_truck_journey_distance_l2768_276895


namespace NUMINAMATH_CALUDE_sum_even_numbers_1_to_200_l2768_276847

/- Define the sum of even numbers from 1 to n -/
def sumEvenNumbers (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

/- Theorem statement -/
theorem sum_even_numbers_1_to_200 : sumEvenNumbers 200 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_numbers_1_to_200_l2768_276847


namespace NUMINAMATH_CALUDE_lilly_fish_count_l2768_276868

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 9

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 19

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := total_fish - rosy_fish

theorem lilly_fish_count : lilly_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_lilly_fish_count_l2768_276868


namespace NUMINAMATH_CALUDE_line_intercept_product_l2768_276832

/-- Given a line 8x + 5y + c = 0, if the product of its x-intercept and y-intercept is 24,
    then c = ±8√15 -/
theorem line_intercept_product (c : ℝ) : 
  (∃ x y : ℝ, 8*x + 5*y + c = 0 ∧ x * y = 24) → 
  (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) := by
sorry

end NUMINAMATH_CALUDE_line_intercept_product_l2768_276832


namespace NUMINAMATH_CALUDE_homes_numbering_twos_l2768_276825

/-- In a city with 100 homes numbered from 1 to 100, 
    the number of 2's used in the numbering is 20. -/
theorem homes_numbering_twos (homes : Nat) (twos_used : Nat) : 
  homes = 100 → twos_used = 20 := by
  sorry

#check homes_numbering_twos

end NUMINAMATH_CALUDE_homes_numbering_twos_l2768_276825


namespace NUMINAMATH_CALUDE_prob_n_minus_one_matches_is_zero_l2768_276869

/-- Represents the number of pairs in the matching problem -/
def n : ℕ := 10

/-- Represents a function that returns the probability of correctly matching
    exactly k pairs out of n pairs when choosing randomly -/
noncomputable def probability_exact_matches (k : ℕ) : ℝ := sorry

/-- Theorem stating that the probability of correctly matching exactly n-1 pairs
    out of n pairs is 0 when choosing randomly -/
theorem prob_n_minus_one_matches_is_zero :
  probability_exact_matches (n - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_prob_n_minus_one_matches_is_zero_l2768_276869


namespace NUMINAMATH_CALUDE_area_between_circles_l2768_276860

/-- The area between a circumscribing circle and two externally tangent circles -/
theorem area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - (π * r₁^2 + π * r₂^2) = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_circles_l2768_276860


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l2768_276829

theorem ufo_convention_attendees :
  let total_attendees : ℕ := 450
  let male_female_difference : ℕ := 26
  let male_attendees : ℕ := (total_attendees + male_female_difference) / 2
  male_attendees = 238 :=
by sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l2768_276829


namespace NUMINAMATH_CALUDE_product_abcd_l2768_276899

/-- Given a system of equations, prove that the product of a, b, c, and d is equal to 58653 / 10716361 -/
theorem product_abcd (a b c d : ℚ) : 
  (4*a + 5*b + 7*c + 9*d = 56) →
  (4*(d+c) = b) →
  (4*b + 2*c = a) →
  (c - 2 = d) →
  a * b * c * d = 58653 / 10716361 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l2768_276899


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l2768_276857

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

-- State the theorem
theorem derivative_f_at_2 : 
  deriv f 2 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l2768_276857


namespace NUMINAMATH_CALUDE_angle_measure_of_special_triangle_l2768_276859

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (angle_range : 0 < C ∧ C < π)
  (side_relation : a^2 + b^2 + a*b = c^2)

-- Theorem statement
theorem angle_measure_of_special_triangle (t : Triangle) : t.C = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_of_special_triangle_l2768_276859


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l2768_276846

def is_valid_sequence (seq : List Nat) : Prop :=
  (∀ i, i + 2 < seq.length → (seq[i]! + seq[i+1]! + seq[i+2]!) % 2 = 0) ∧
  (∀ i, i + 3 < seq.length → (seq[i]! + seq[i+1]! + seq[i+2]! + seq[i+3]!) % 2 = 1)

theorem max_valid_sequence_length :
  (∃ (seq : List Nat), is_valid_sequence seq ∧ seq.length = 5) ∧
  (∀ (seq : List Nat), is_valid_sequence seq → seq.length ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l2768_276846


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2768_276823

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a < -1 → a^2 - 5*a - 6 > 0) ∧
  (∃ a, a^2 - 5*a - 6 > 0 ∧ a ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2768_276823


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l2768_276844

def S : Finset Int := {-5, 30, -2, 15, -4}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x + y + z = -11 ∧
  ∀ (d e f : Int), d ∈ S → e ∈ S → f ∈ S → 
  d ≠ e ∧ e ≠ f ∧ d ≠ f → 
  d + e + f ≥ -11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l2768_276844


namespace NUMINAMATH_CALUDE_katies_first_stopover_l2768_276836

/-- Calculates the distance to the first stopover given the total distance,
    distance to the second stopover, and additional distance to the final destination -/
def distance_to_first_stopover (total_distance : ℕ) (second_stopover : ℕ) (additional_distance : ℕ) : ℕ :=
  second_stopover - (total_distance - second_stopover - additional_distance)

/-- Proves that given the specific distances in Katie's trip,
    the distance to the first stopover is 104 miles -/
theorem katies_first_stopover :
  distance_to_first_stopover 436 236 68 = 104 := by
  sorry

#eval distance_to_first_stopover 436 236 68

end NUMINAMATH_CALUDE_katies_first_stopover_l2768_276836


namespace NUMINAMATH_CALUDE_ellen_lego_count_l2768_276879

/-- Calculates the final number of legos Ellen has after a series of transactions -/
def final_lego_count (initial : ℕ) : ℕ :=
  let after_week1 := initial - initial / 5
  let after_week2 := after_week1 + after_week1 / 4
  let after_week3 := after_week2 - 57
  after_week3 + after_week3 / 10

/-- Theorem stating that Ellen ends up with 355 legos -/
theorem ellen_lego_count : final_lego_count 380 = 355 := by
  sorry


end NUMINAMATH_CALUDE_ellen_lego_count_l2768_276879


namespace NUMINAMATH_CALUDE_largest_sphere_in_cone_l2768_276805

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a circle in the xy plane -/
structure Circle where
  center : Point3D
  radius : ℝ

/-- Represents a cone with circular base and vertex -/
structure Cone where
  base : Circle
  vertex : Point3D

/-- The largest possible radius of a sphere contained in a cone -/
def largestSphereRadius (cone : Cone) : ℝ :=
  sorry

theorem largest_sphere_in_cone :
  let c : Circle := { center := ⟨0, 0, 0⟩, radius := 1 }
  let p : Point3D := ⟨3, 4, 8⟩
  let cone : Cone := { base := c, vertex := p }
  largestSphereRadius cone = 3 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_sphere_in_cone_l2768_276805


namespace NUMINAMATH_CALUDE_problem_solution_l2768_276826

theorem problem_solution (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 
  (a = Real.sqrt 2 + 1) ∧ 
  (a^2 - 2*a = 1) ∧ 
  (2*a^3 - 4*a^2 - 1 = 2 * Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2768_276826


namespace NUMINAMATH_CALUDE_disk_color_difference_l2768_276865

theorem disk_color_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) :
  total = 144 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  let total_ratio := blue_ratio + yellow_ratio + green_ratio
  let disks_per_part := total / total_ratio
  let blue_disks := blue_ratio * disks_per_part
  let green_disks := green_ratio * disks_per_part
  green_disks - blue_disks = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_disk_color_difference_l2768_276865


namespace NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l2768_276839

/-- The points of intersection of the given lines form a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∀ (s x y : ℝ), 
    (2*s*x - 3*y - 5*s = 0) → 
    (2*x - 3*s*y + 4 = 0) → 
    ∃ (a b : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l2768_276839


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2768_276898

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) 
  (inradius : ℝ) 
  (h_perimeter : perimeter = 42) 
  (h_inradius : inradius = 5) : 
  inradius * (perimeter / 2) = 105 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2768_276898


namespace NUMINAMATH_CALUDE_potato_cost_for_group_l2768_276864

/-- The cost of potatoes for a group, given the number of people, amount each person eats,
    bag size, and cost per bag. -/
def potatoCost (people : ℕ) (poundsPerPerson : ℚ) (bagSize : ℕ) (costPerBag : ℚ) : ℚ :=
  let totalPounds : ℚ := people * poundsPerPerson
  let bagsNeeded : ℕ := (totalPounds / bagSize).ceil.toNat
  bagsNeeded * costPerBag

/-- Theorem stating that the cost of potatoes for 40 people, where each person eats 1.5 pounds,
    and a 20-pound bag costs $5, is $15. -/
theorem potato_cost_for_group : potatoCost 40 (3/2) 20 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_potato_cost_for_group_l2768_276864


namespace NUMINAMATH_CALUDE_sprint_stats_change_l2768_276841

theorem sprint_stats_change (n : Nat) (avg_10 : ℝ) (var_10 : ℝ) (time_11 : ℝ) :
  n = 10 →
  avg_10 = 8.2 →
  var_10 = 2.2 →
  time_11 = 8.2 →
  let avg_11 := (n * avg_10 + time_11) / (n + 1)
  let var_11 := (n * var_10 + (time_11 - avg_10)^2) / (n + 1)
  avg_11 = avg_10 ∧ var_11 < var_10 := by
  sorry

#check sprint_stats_change

end NUMINAMATH_CALUDE_sprint_stats_change_l2768_276841


namespace NUMINAMATH_CALUDE_children_who_got_on_bus_l2768_276885

/-- Proves the number of children who got on the bus -/
theorem children_who_got_on_bus 
  (initial_children : ℕ) 
  (children_who_got_off : ℕ) 
  (final_children : ℕ) 
  (h1 : initial_children = 21)
  (h2 : children_who_got_off = 10)
  (h3 : final_children = 16)
  : initial_children - children_who_got_off + (final_children - (initial_children - children_who_got_off)) = final_children ∧ 
    final_children - (initial_children - children_who_got_off) = 5 :=
by sorry

end NUMINAMATH_CALUDE_children_who_got_on_bus_l2768_276885
