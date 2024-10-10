import Mathlib

namespace max_marble_diff_is_six_l704_70439

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : Nat
  color2 : String
  count2 : Nat

/-- Calculates the absolute difference between two numbers -/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Theorem: The maximum difference between marble counts in any basket is 6 -/
theorem max_marble_diff_is_six (basketA basketB basketC : Basket)
  (hA : basketA = { color1 := "red", count1 := 4, color2 := "yellow", count2 := 2 })
  (hB : basketB = { color1 := "green", count1 := 6, color2 := "yellow", count2 := 1 })
  (hC : basketC = { color1 := "white", count1 := 3, color2 := "yellow", count2 := 9 }) :
  max (absDiff basketA.count1 basketA.count2)
      (max (absDiff basketB.count1 basketB.count2)
           (absDiff basketC.count1 basketC.count2)) = 6 := by
  sorry


end max_marble_diff_is_six_l704_70439


namespace trigonometric_identity_l704_70498

theorem trigonometric_identity (t : ℝ) (h : 3 * Real.cos (2 * t) - Real.sin (2 * t) ≠ 0) :
  (6 * Real.cos (2 * t)^3 + 2 * Real.sin (2 * t)^3) / (3 * Real.cos (2 * t) - Real.sin (2 * t))
  = Real.cos (4 * t) := by
  sorry

end trigonometric_identity_l704_70498


namespace work_duration_l704_70410

/-- Given workers A and B with their individual work rates and the time B takes to finish after A leaves,
    prove that A and B worked together for 2 days. -/
theorem work_duration (a_rate b_rate : ℚ) (b_finish_time : ℚ) : 
  a_rate = 1/4 →
  b_rate = 1/10 →
  b_finish_time = 3 →
  ∃ (x : ℚ), x = 2 ∧ (a_rate + b_rate) * x + b_rate * b_finish_time = 1 := by
  sorry

#eval (1/4 : ℚ) + (1/10 : ℚ)  -- Combined work rate
#eval ((1/4 : ℚ) + (1/10 : ℚ)) * 2 + (1/10 : ℚ) * 3  -- Total work done

end work_duration_l704_70410


namespace total_songs_bought_l704_70486

theorem total_songs_bought (country_albums pop_albums rock_albums : ℕ)
  (country_songs_per_album pop_songs_per_album rock_songs_per_album : ℕ) :
  country_albums = 2 ∧
  pop_albums = 8 ∧
  rock_albums = 5 ∧
  country_songs_per_album = 7 ∧
  pop_songs_per_album = 10 ∧
  rock_songs_per_album = 12 →
  country_albums * country_songs_per_album +
  pop_albums * pop_songs_per_album +
  rock_albums * rock_songs_per_album = 154 :=
by sorry

end total_songs_bought_l704_70486


namespace area_bisecting_line_property_l704_70499

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ := (0, 10)
  Q : ℝ × ℝ := (3, 0)
  R : ℝ × ℝ := (9, 0)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Function to check if a line bisects the area of the triangle -/
def bisects_area (t : Triangle) (l : Line) : Prop :=
  sorry -- Definition of area bisection

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  sorry -- Definition of line passing through a point

/-- Theorem stating the property of the area-bisecting line -/
theorem area_bisecting_line_property (t : Triangle) :
  ∃ l : Line, bisects_area t l ∧ passes_through l t.Q ∧ l.slope + l.y_intercept = -20/3 :=
sorry

end area_bisecting_line_property_l704_70499


namespace max_sequence_sum_l704_70435

def arithmetic_sequence (n : ℕ) : ℚ := 5 - (5/7) * (n - 1)

def sequence_sum (n : ℕ) : ℚ := n * (2 * 5 + (n - 1) * (-5/7)) / 2

theorem max_sequence_sum :
  (∃ n : ℕ, sequence_sum n = 20) ∧
  (∀ m : ℕ, sequence_sum m ≤ 20) ∧
  (∀ n : ℕ, sequence_sum n = 20 → (n = 7 ∨ n = 8)) :=
sorry

end max_sequence_sum_l704_70435


namespace horner_v3_equals_108_l704_70400

def horner_v (coeffs : List ℝ) (x : ℝ) : List ℝ :=
  coeffs.scanl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem horner_v3_equals_108 :
  let coeffs := [2, -5, -4, 3, -6, 7]
  let x := 5
  let v := horner_v coeffs x
  v[3] = 108 := by sorry

end horner_v3_equals_108_l704_70400


namespace quilt_shaded_half_l704_70404

/-- Represents a square quilt composed of unit squares -/
structure Quilt :=
  (size : ℕ)
  (shaded_rows : ℕ)

/-- The fraction of the quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  q.shaded_rows / q.size

/-- Theorem: For a 4x4 quilt with 2 shaded rows, the shaded fraction is 1/2 -/
theorem quilt_shaded_half (q : Quilt) 
  (h1 : q.size = 4) 
  (h2 : q.shaded_rows = 2) : 
  shaded_fraction q = 1/2 := by
sorry

end quilt_shaded_half_l704_70404


namespace integral_f_cos_nonnegative_l704_70482

open MeasureTheory Interval RealInnerProductSpace Set

theorem integral_f_cos_nonnegative 
  (f : ℝ → ℝ) 
  (hf_continuous : ContinuousOn f (Icc 0 (2 * Real.pi)))
  (hf'_continuous : ContinuousOn (deriv f) (Icc 0 (2 * Real.pi)))
  (hf''_continuous : ContinuousOn (deriv^[2] f) (Icc 0 (2 * Real.pi)))
  (hf''_nonneg : ∀ x ∈ Icc 0 (2 * Real.pi), deriv^[2] f x ≥ 0) :
  ∫ x in Icc 0 (2 * Real.pi), f x * Real.cos x ≥ 0 := by
  sorry

end integral_f_cos_nonnegative_l704_70482


namespace line_ellipse_intersection_l704_70413

theorem line_ellipse_intersection (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 7) ↔ m^2 ≥ (9/50) := by
sorry

end line_ellipse_intersection_l704_70413


namespace letter_cost_l704_70453

/-- The cost to mail each letter, given the total cost, package cost, and number of letters and packages. -/
theorem letter_cost (total_cost package_cost : ℚ) (num_letters num_packages : ℕ) : 
  total_cost = 4.49 →
  package_cost = 0.88 →
  num_letters = 5 →
  num_packages = 3 →
  (num_letters : ℚ) * ((total_cost - (package_cost * (num_packages : ℚ))) / (num_letters : ℚ)) = 0.37 := by
  sorry

end letter_cost_l704_70453


namespace max_servings_is_ten_l704_70436

/-- Represents the number of chunks per serving for each fruit type -/
structure FruitRatio where
  cantaloupe : ℕ
  honeydew : ℕ
  pineapple : ℕ
  watermelon : ℕ

/-- Represents the available chunks of each fruit type -/
structure AvailableFruit where
  cantaloupe : ℕ
  honeydew : ℕ
  pineapple : ℕ
  watermelon : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (ratio : FruitRatio) (available : AvailableFruit) : ℕ :=
  min
    (available.cantaloupe / ratio.cantaloupe)
    (min
      (available.honeydew / ratio.honeydew)
      (min
        (available.pineapple / ratio.pineapple)
        (available.watermelon / ratio.watermelon)))

/-- The given fruit ratio -/
def givenRatio : FruitRatio :=
  { cantaloupe := 3
  , honeydew := 2
  , pineapple := 1
  , watermelon := 4 }

/-- The available fruit chunks -/
def givenAvailable : AvailableFruit :=
  { cantaloupe := 30
  , honeydew := 42
  , pineapple := 12
  , watermelon := 56 }

theorem max_servings_is_ten :
  maxServings givenRatio givenAvailable = 10 := by
  sorry


end max_servings_is_ten_l704_70436


namespace binomial_coefficient_equation_solution_l704_70424

theorem binomial_coefficient_equation_solution (x : ℕ) : 
  Nat.choose 11 x = Nat.choose 11 (2*x - 4) ↔ x = 4 ∨ x = 5 := by
  sorry

end binomial_coefficient_equation_solution_l704_70424


namespace absolute_difference_of_product_and_sum_l704_70491

theorem absolute_difference_of_product_and_sum (m n : ℝ) 
  (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 := by
  sorry

end absolute_difference_of_product_and_sum_l704_70491


namespace coconut_grove_problem_l704_70443

theorem coconut_grove_problem (x : ℝ) : 
  (((x + 2) * 40 + x * 120 + (x - 2) * 180) / (3 * x) = 100) → x = 7 := by
  sorry

end coconut_grove_problem_l704_70443


namespace cat_food_finished_l704_70490

def daily_consumption : ℚ := 1/4 + 1/6

def total_cans : ℕ := 10

def days_to_finish : ℕ := 15

theorem cat_food_finished :
  (daily_consumption * days_to_finish : ℚ) ≥ total_cans ∧
  (daily_consumption * (days_to_finish - 1) : ℚ) < total_cans := by
  sorry

end cat_food_finished_l704_70490


namespace shaded_area_of_carpet_l704_70407

/-- Given a square carpet with the following properties:
  * Side length of the carpet is 12 feet
  * Contains one large shaded square and twelve smaller congruent shaded squares
  * S is the side length of the large shaded square
  * T is the side length of each smaller shaded square
  * The ratio 12:S is 4
  * The ratio S:T is 4
  Prove that the total shaded area is 15.75 square feet -/
theorem shaded_area_of_carpet (S T : ℝ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 4)
  : S^2 + 12 * T^2 = 15.75 := by
  sorry

end shaded_area_of_carpet_l704_70407


namespace circle_iff_a_eq_neg_one_l704_70422

/-- Represents a quadratic equation in x and y with parameter a -/
def is_circle (a : ℝ) : Prop :=
  ∃ h k r, ∀ x y : ℝ, 
    a^2 * x^2 + (a + 2) * y^2 + 2*a*x + a = 0 ↔ 
    (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

/-- The equation represents a circle if and only if a = -1 -/
theorem circle_iff_a_eq_neg_one :
  ∀ a : ℝ, is_circle a ↔ a = -1 := by sorry

end circle_iff_a_eq_neg_one_l704_70422


namespace area_between_curves_l704_70475

-- Define the functions
def f (x : ℝ) : ℝ := x
def g (x : ℝ) : ℝ := 2 - x^2

-- Define the intersection points
def x₁ : ℝ := -2
def x₂ : ℝ := 1

-- State the theorem
theorem area_between_curves : 
  (∫ (x : ℝ) in x₁..x₂, g x - f x) = 9/2 := by sorry

end area_between_curves_l704_70475


namespace perfect_linear_correlation_l704_70457

/-- A scatter plot where all points fall on a straight line -/
structure PerfectLinearScatterPlot where
  /-- The slope of the line (non-zero real number) -/
  slope : ℝ
  /-- Assumption that the slope is non-zero -/
  slope_nonzero : slope ≠ 0

/-- The correlation coefficient R^2 for a scatter plot -/
def correlation_coefficient (plot : PerfectLinearScatterPlot) : ℝ := sorry

/-- Theorem: The correlation coefficient R^2 is 1 for a perfect linear scatter plot -/
theorem perfect_linear_correlation 
  (plot : PerfectLinearScatterPlot) : 
  correlation_coefficient plot = 1 := by sorry

end perfect_linear_correlation_l704_70457


namespace total_water_flow_l704_70463

def water_flow_rate : ℚ := 2 + 2/3
def time_period : ℕ := 9

theorem total_water_flow (rate : ℚ) (time : ℕ) (h1 : rate = water_flow_rate) (h2 : time = time_period) :
  rate * time = 24 := by
  sorry

end total_water_flow_l704_70463


namespace periodic_function_value_l704_70474

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2015 = 5 → f 2016 = 3 := by
  sorry

end periodic_function_value_l704_70474


namespace unique_solution_ceiling_equation_l704_70451

theorem unique_solution_ceiling_equation :
  ∃! c : ℝ, c + ⌈c⌉ = 23.2 :=
by
  sorry

end unique_solution_ceiling_equation_l704_70451


namespace rectangle_area_perimeter_relation_l704_70481

theorem rectangle_area_perimeter_relation :
  ∀ (a b : ℕ), 
    a ≠ b →                  -- non-square condition
    a > 0 →                  -- positive dimension
    b > 0 →                  -- positive dimension
    a * b = 2 * (2 * a + 2 * b) →  -- area equals twice perimeter
    2 * (a + b) = 36 :=      -- perimeter is 36
by
  sorry

end rectangle_area_perimeter_relation_l704_70481


namespace page_number_added_twice_l704_70408

theorem page_number_added_twice (m : ℕ) (p : ℕ) : 
  m = 71 → 
  1 ≤ p → 
  p ≤ m → 
  (m * (m + 1)) / 2 + p = 2550 → 
  p = 6 := by
sorry

end page_number_added_twice_l704_70408


namespace percent_of_a_l704_70468

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10/3) * a := by
  sorry

end percent_of_a_l704_70468


namespace proj_scale_proj_add_l704_70495

-- Define the 2D vector type
def Vector2D := ℝ × ℝ

-- Define the projection operation on x-axis
def proj_x (v : Vector2D) : ℝ := v.1

-- Define the projection operation on y-axis
def proj_y (v : Vector2D) : ℝ := v.2

-- Define vector addition
def add (u v : Vector2D) : Vector2D := (u.1 + v.1, u.2 + v.2)

-- Define scalar multiplication
def scale (k : ℝ) (v : Vector2D) : Vector2D := (k * v.1, k * v.2)

-- Theorem for property 1 (scalar multiplication)
theorem proj_scale (k : ℝ) (v : Vector2D) :
  proj_x (scale k v) = k * proj_x v ∧ proj_y (scale k v) = k * proj_y v := by
  sorry

-- Theorem for property 2 (vector addition)
theorem proj_add (u v : Vector2D) :
  proj_x (add u v) = proj_x u + proj_x v ∧ proj_y (add u v) = proj_y u + proj_y v := by
  sorry

end proj_scale_proj_add_l704_70495


namespace line_parallel_perpendicular_plane_l704_70444

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

/-- Two geometric objects are different -/
def different (a b : α) : Prop := a ≠ b

theorem line_parallel_perpendicular_plane 
  (m n : Line) (α : Plane) 
  (h1 : different m n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n α) : 
  perpendicular m α := by sorry

end line_parallel_perpendicular_plane_l704_70444


namespace marbles_given_to_eric_l704_70462

def marble_redistribution (tyrone_initial : ℕ) (eric_initial : ℕ) (x : ℕ) : Prop :=
  (tyrone_initial - x) = 3 * (eric_initial + x)

theorem marbles_given_to_eric 
  (tyrone_initial : ℕ) (eric_initial : ℕ) (x : ℕ)
  (h1 : tyrone_initial = 120)
  (h2 : eric_initial = 20)
  (h3 : marble_redistribution tyrone_initial eric_initial x) :
  x = 15 := by
  sorry

end marbles_given_to_eric_l704_70462


namespace equal_candy_sharing_l704_70416

/-- Represents the number of candies each person has initially -/
structure CandyDistribution :=
  (mark : ℕ)
  (peter : ℕ)
  (john : ℕ)

/-- Calculates the total number of candies -/
def totalCandies (d : CandyDistribution) : ℕ :=
  d.mark + d.peter + d.john

/-- Calculates the number of candies each person gets after equal sharing -/
def sharedCandies (d : CandyDistribution) : ℕ :=
  totalCandies d / 3

/-- Proves that when Mark (30 candies), Peter (25 candies), and John (35 candies)
    combine their candies and share equally, each person will have 30 candies -/
theorem equal_candy_sharing :
  let d : CandyDistribution := { mark := 30, peter := 25, john := 35 }
  sharedCandies d = 30 := by
  sorry

end equal_candy_sharing_l704_70416


namespace otimes_neg_two_four_otimes_equation_l704_70484

/-- Define the ⊗ operation for rational numbers -/
def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

/-- Theorem 1: (-2) ⊗ 4 = -50 -/
theorem otimes_neg_two_four : otimes (-2) 4 = -50 := by sorry

/-- Theorem 2: If x ⊗ 3 = y ⊗ (-3), then 8x - 2y + 5 = 5 -/
theorem otimes_equation (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 
  8 * x - 2 * y + 5 = 5 := by sorry

end otimes_neg_two_four_otimes_equation_l704_70484


namespace vector_problem_l704_70464

/-- Given vectors and triangle properties, prove vector n coordinates and magnitude range of n + p -/
theorem vector_problem (m n p q : ℝ × ℝ) (A B C : ℝ) : 
  m = (1, 1) →
  q = (1, 0) →
  (m.1 * n.1 + m.2 * n.2) = -1 →
  ∃ (k : ℝ), n = k • q →
  p = (2 * Real.cos (C / 2) ^ 2, Real.cos A) →
  B = π / 3 →
  A + B + C = π →
  (n = (-1, 0) ∧ Real.sqrt 2 / 2 ≤ Real.sqrt ((n.1 + p.1)^2 + (n.2 + p.2)^2) ∧ 
   Real.sqrt ((n.1 + p.1)^2 + (n.2 + p.2)^2) < Real.sqrt 5 / 2) :=
by sorry

end vector_problem_l704_70464


namespace money_distribution_l704_70423

def total_proportion : ℕ := 5 + 2 + 4 + 3

theorem money_distribution (S : ℚ) (A_share B_share C_share D_share : ℚ) : 
  A_share = 2500 ∧ 
  A_share = (5 : ℚ) / total_proportion * S ∧
  B_share = (2 : ℚ) / total_proportion * S ∧
  C_share = (4 : ℚ) / total_proportion * S ∧
  D_share = (3 : ℚ) / total_proportion * S →
  C_share - D_share = 500 := by
sorry

end money_distribution_l704_70423


namespace raisin_count_l704_70447

theorem raisin_count (total_raisins : ℕ) (total_boxes : ℕ) (second_box : ℕ) (other_boxes : ℕ) (other_box_count : ℕ) :
  total_raisins = 437 →
  total_boxes = 5 →
  second_box = 74 →
  other_boxes = 97 →
  other_box_count = 3 →
  ∃ (first_box : ℕ), first_box = total_raisins - (second_box + other_box_count * other_boxes) ∧ first_box = 72 :=
by
  sorry

end raisin_count_l704_70447


namespace one_fourths_in_five_thirds_l704_70497

theorem one_fourths_in_five_thirds : (5 : ℚ) / 3 / (1 / 4) = 20 / 3 := by
  sorry

end one_fourths_in_five_thirds_l704_70497


namespace quadratic_sequence_existence_l704_70479

theorem quadratic_sequence_existence (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧
  ∀ i : ℕ, i ≤ n → i ≠ 0 → |a i - a (i - 1)| = i^2 :=
sorry

end quadratic_sequence_existence_l704_70479


namespace least_pennies_count_l704_70417

theorem least_pennies_count (a : ℕ) : 
  (a > 0) → 
  (a % 7 = 1) → 
  (a % 3 = 0) → 
  (∀ b : ℕ, b > 0 → b % 7 = 1 → b % 3 = 0 → a ≤ b) → 
  a = 15 := by
sorry

end least_pennies_count_l704_70417


namespace maria_earnings_l704_70470

def brush_cost_1 : ℕ := 20
def brush_cost_2 : ℕ := 25
def brush_cost_3 : ℕ := 30
def acrylic_paint_cost : ℕ := 8
def oil_paint_cost : ℕ := 12
def acrylic_paint_amount : ℕ := 5
def oil_paint_amount : ℕ := 3
def selling_price : ℕ := 200

def total_brush_cost : ℕ := brush_cost_1 + brush_cost_2 + brush_cost_3

def canvas_cost_1 : ℕ := 3 * total_brush_cost
def canvas_cost_2 : ℕ := 2 * total_brush_cost

def total_paint_cost : ℕ := acrylic_paint_cost * acrylic_paint_amount + oil_paint_cost * oil_paint_amount

def total_cost : ℕ := total_brush_cost + canvas_cost_1 + canvas_cost_2 + total_paint_cost

theorem maria_earnings : (selling_price : ℤ) - total_cost = -326 := by sorry

end maria_earnings_l704_70470


namespace min_sum_squared_distances_l704_70473

/-- Given collinear points A, B, C, D, and E with specified distances between them,
    this function calculates the sum of squared distances from these points to a point P on AE. -/
def sum_of_squared_distances (x : ℝ) : ℝ :=
  x^2 + (x - 2)^2 + (x - 4)^2 + (x - 7)^2 + (x - 11)^2

/-- The theorem states that the minimum value of the sum of squared distances
    from points A, B, C, D, and E to any point P on line segment AE is 54.8,
    given the specified distances between the points. -/
theorem min_sum_squared_distances :
  ∃ (min_value : ℝ), min_value = 54.8 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 11 → sum_of_squared_distances x ≥ min_value :=
sorry

end min_sum_squared_distances_l704_70473


namespace eq1_roots_eq2_roots_l704_70460

-- Define the quadratic equations
def eq1 (x : ℝ) : Prop := x^2 + 10*x + 16 = 0
def eq2 (x : ℝ) : Prop := x*(x+4) = 8*x + 12

-- Theorem for the first equation
theorem eq1_roots : 
  (∃ x : ℝ, eq1 x) ↔ (eq1 (-2) ∧ eq1 (-8)) :=
sorry

-- Theorem for the second equation
theorem eq2_roots : 
  (∃ x : ℝ, eq2 x) ↔ (eq2 (-2) ∧ eq2 6) :=
sorry

end eq1_roots_eq2_roots_l704_70460


namespace salary_average_increase_l704_70415

theorem salary_average_increase 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 24 → 
  avg_salary = 2400 → 
  manager_salary = 4900 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 100 := by
  sorry

end salary_average_increase_l704_70415


namespace juan_running_time_l704_70465

theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 250) (h2 : speed = 8) :
  distance / speed = 31.25 := by
  sorry

end juan_running_time_l704_70465


namespace fourth_sampled_number_l704_70432

/-- Represents a random number table -/
def RandomNumberTable := List (List Nat)

/-- Represents a position in the random number table -/
structure TablePosition where
  row : Nat
  column : Nat

/-- Checks if a number is valid for sampling (between 1 and 40) -/
def isValidNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 40

/-- Gets the next position in the table -/
def nextPosition (pos : TablePosition) (tableWidth : Nat) : TablePosition :=
  if pos.column < tableWidth then
    { row := pos.row, column := pos.column + 1 }
  else
    { row := pos.row + 1, column := 1 }

/-- Samples the next valid number from the table -/
def sampleNextNumber (table : RandomNumberTable) (startPos : TablePosition) : Option Nat :=
  sorry

/-- Samples n valid numbers from the table -/
def sampleNumbers (table : RandomNumberTable) (startPos : TablePosition) (n : Nat) : List Nat :=
  sorry

/-- The main theorem to prove -/
theorem fourth_sampled_number
  (table : RandomNumberTable)
  (startPos : TablePosition)
  (h_table : table = [
    [84, 42, 17, 56, 31, 07, 23, 55, 06, 82, 77, 04, 74, 43, 59, 76, 30, 63, 50, 25, 83, 92, 12, 06],
    [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38]
  ])
  (h_startPos : startPos = { row := 0, column := 7 })
  : (sampleNumbers table startPos 4).get! 3 = 6 := by
  sorry

end fourth_sampled_number_l704_70432


namespace angle_terminal_side_value_l704_70431

/-- Given that the terminal side of angle α passes through the point P(-4a,3a) where a ≠ 0,
    the value of 2sin α + cos α is either 2/5 or -2/5 -/
theorem angle_terminal_side_value (a : ℝ) (α : ℝ) (h : a ≠ 0) :
  let P : ℝ × ℝ := (-4 * a, 3 * a)
  (∃ t : ℝ, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5 :=
by sorry

end angle_terminal_side_value_l704_70431


namespace luisas_books_l704_70419

theorem luisas_books (maddie_books amy_books : ℕ) (h1 : maddie_books = 15) (h2 : amy_books = 6)
  (h3 : ∃ luisa_books : ℕ, amy_books + luisa_books = maddie_books + 9) :
  ∃ luisa_books : ℕ, luisa_books = 18 := by
  sorry

end luisas_books_l704_70419


namespace modular_inverse_of_5_mod_31_l704_70461

theorem modular_inverse_of_5_mod_31 :
  ∃ x : ℕ, x < 31 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end modular_inverse_of_5_mod_31_l704_70461


namespace expression_evaluation_l704_70454

theorem expression_evaluation : 
  let f (x : ℝ) := (x + 1) / (x - 1)
  let g (x : ℝ) := (f x + 1) / (f x - 1)
  g (1/2) = -3 := by sorry

end expression_evaluation_l704_70454


namespace sum_of_largest_and_smallest_l704_70425

/-- A function that generates all valid eight-digit numbers using the digits 4, 0, 2, 6 twice each -/
def validNumbers : List Nat := sorry

/-- The largest eight-digit number that can be formed using the digits 4, 0, 2, 6 twice each -/
def largestNumber : Nat := sorry

/-- The smallest eight-digit number that can be formed using the digits 4, 0, 2, 6 twice each -/
def smallestNumber : Nat := sorry

/-- Theorem stating that the sum of the largest and smallest valid numbers is 86,466,666 -/
theorem sum_of_largest_and_smallest :
  largestNumber + smallestNumber = 86466666 := by sorry

end sum_of_largest_and_smallest_l704_70425


namespace recycling_money_calculation_l704_70440

/-- Calculates the total money received from recycling cans and newspapers. -/
def recycling_money (can_rate : ℚ) (newspaper_rate : ℚ) (cans : ℕ) (newspapers : ℕ) : ℚ :=
  (can_rate * (cans / 12 : ℚ)) + (newspaper_rate * (newspapers / 5 : ℚ))

/-- Theorem: Given the recycling rates and the family's collection, the total money received is $12. -/
theorem recycling_money_calculation :
  recycling_money (1/2) (3/2) 144 20 = 12 := by
  sorry

end recycling_money_calculation_l704_70440


namespace piggy_bank_dimes_l704_70487

theorem piggy_bank_dimes (total_value : ℚ) (total_coins : ℕ) 
  (quarter_value : ℚ) (dime_value : ℚ) :
  total_value = 39.5 ∧ 
  total_coins = 200 ∧ 
  quarter_value = 0.25 ∧ 
  dime_value = 0.1 →
  ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧
    quarter_value * quarters + dime_value * dimes = total_value ∧
    dimes = 70 := by
  sorry

end piggy_bank_dimes_l704_70487


namespace complex_modulus_range_l704_70449

theorem complex_modulus_range (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (x : ℝ), x = Complex.abs ((z - 2) * (z + 1)^2) ∧ 0 ≤ x ∧ x ≤ 3 * Real.sqrt 3 :=
by sorry

end complex_modulus_range_l704_70449


namespace circle_with_n_integer_points_l704_70412

/-- A circle in the Euclidean plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of integer points on a circle -/
def num_integer_points (c : Circle) : ℕ :=
  sorry

/-- For any natural number n, there exists a circle with exactly n integer points -/
theorem circle_with_n_integer_points :
  ∀ n : ℕ, ∃ c : Circle, num_integer_points c = n :=
sorry

end circle_with_n_integer_points_l704_70412


namespace unanswered_test_completion_ways_l704_70402

/-- Represents a multiple-choice test -/
structure MultipleChoiceTest where
  num_questions : ℕ
  choices_per_question : ℕ

/-- Calculates the number of ways to complete the test with all questions unanswered -/
def ways_to_complete_unanswered (test : MultipleChoiceTest) : ℕ := 1

/-- Theorem: For a 4-question test with 5 choices per question, there is only one way to complete it with all questions unanswered -/
theorem unanswered_test_completion_ways 
  (test : MultipleChoiceTest) 
  (h1 : test.num_questions = 4) 
  (h2 : test.choices_per_question = 5) : 
  ways_to_complete_unanswered test = 1 := by
  sorry

end unanswered_test_completion_ways_l704_70402


namespace max_value_of_f_sum_of_powers_gt_one_l704_70488

-- Part 1
theorem max_value_of_f (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ M : ℝ, M = 1 ∧ ∀ x > -1, (1 + x)^a - a * x ≤ M := by sorry

-- Part 2
theorem sum_of_powers_gt_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^b + b^a > 1 := by sorry

end max_value_of_f_sum_of_powers_gt_one_l704_70488


namespace similar_triangles_side_length_l704_70442

/-- Given two similar triangles PQR and STU, prove that PQ = 10.5 -/
theorem similar_triangles_side_length 
  (PQ ST PR SU : ℝ) 
  (h_similar : PQ / ST = PR / SU) 
  (h_ST : ST = 4.5) 
  (h_PR : PR = 21) 
  (h_SU : SU = 9) : 
  PQ = 10.5 := by
  sorry

end similar_triangles_side_length_l704_70442


namespace pure_imaginary_complex_number_l704_70477

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a^2 - 4 + (a^2 - 3*a + 2)*I
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by sorry

end pure_imaginary_complex_number_l704_70477


namespace floor_problem_solution_exists_and_unique_floor_length_satisfies_conditions_l704_70429

/-- Represents the dimensions and costs of a rectangular floor with a painted border. -/
structure FloorProblem where
  breadth : ℝ
  length_ratio : ℝ
  floor_paint_rate : ℝ
  floor_paint_cost : ℝ
  border_paint_rate : ℝ
  total_paint_cost : ℝ

/-- The main theorem stating the existence and uniqueness of a solution to the floor problem. -/
theorem floor_problem_solution_exists_and_unique :
  ∃! (fp : FloorProblem),
    fp.length_ratio = 3 ∧
    fp.floor_paint_rate = 3.00001 ∧
    fp.floor_paint_cost = 361 ∧
    fp.border_paint_rate = 15 ∧
    fp.total_paint_cost = 500 ∧
    fp.floor_paint_rate * (fp.length_ratio * fp.breadth * fp.breadth) = fp.floor_paint_cost ∧
    fp.border_paint_rate * (2 * (fp.length_ratio * fp.breadth + fp.breadth)) + fp.floor_paint_cost = fp.total_paint_cost :=
  sorry

/-- Function to calculate the length of the floor given a FloorProblem instance. -/
def calculate_floor_length (fp : FloorProblem) : ℝ :=
  fp.length_ratio * fp.breadth

/-- Theorem stating that the calculated floor length satisfies the problem conditions. -/
theorem floor_length_satisfies_conditions (fp : FloorProblem) :
  fp.length_ratio = 3 →
  fp.floor_paint_rate = 3.00001 →
  fp.floor_paint_cost = 361 →
  fp.border_paint_rate = 15 →
  fp.total_paint_cost = 500 →
  fp.floor_paint_rate * (fp.length_ratio * fp.breadth * fp.breadth) = fp.floor_paint_cost →
  fp.border_paint_rate * (2 * (fp.length_ratio * fp.breadth + fp.breadth)) + fp.floor_paint_cost = fp.total_paint_cost →
  ∃ (length : ℝ), length = calculate_floor_length fp :=
  sorry

end floor_problem_solution_exists_and_unique_floor_length_satisfies_conditions_l704_70429


namespace least_integer_in_ratio_l704_70437

theorem least_integer_in_ratio (a b c : ℕ+) : 
  (a : ℚ) + (b : ℚ) + (c : ℚ) = 90 →
  (b : ℚ) = 2 * (a : ℚ) →
  (c : ℚ) = 5 * (a : ℚ) →
  (a : ℚ) = 45 / 4 :=
by sorry

end least_integer_in_ratio_l704_70437


namespace first_term_of_ap_l704_70459

/-- 
Given an arithmetic progression where:
- The 10th term is 26
- The common difference is 2

Prove that the first term is 8
-/
theorem first_term_of_ap (a : ℝ) : 
  (∃ (d : ℝ), d = 2 ∧ a + 9 * d = 26) → a = 8 := by
  sorry

end first_term_of_ap_l704_70459


namespace b_is_negative_l704_70480

def is_two_positive_two_negative (a b : ℝ) : Prop :=
  (((a + b > 0) ∧ (a - b > 0)) ∨ ((a + b > 0) ∧ (a * b > 0)) ∨ ((a + b > 0) ∧ (a / b > 0)) ∨
   ((a - b > 0) ∧ (a * b > 0)) ∨ ((a - b > 0) ∧ (a / b > 0)) ∨ ((a * b > 0) ∧ (a / b > 0))) ∧
  (((a + b < 0) ∧ (a - b < 0)) ∨ ((a + b < 0) ∧ (a * b < 0)) ∨ ((a + b < 0) ∧ (a / b < 0)) ∨
   ((a - b < 0) ∧ (a * b < 0)) ∨ ((a - b < 0) ∧ (a / b < 0)) ∨ ((a * b < 0) ∧ (a / b < 0)))

theorem b_is_negative (a b : ℝ) (h : b ≠ 0) (condition : is_two_positive_two_negative a b) : b < 0 := by
  sorry

end b_is_negative_l704_70480


namespace intercept_sum_l704_70411

/-- A line is described by the equation y + 3 = -3(x - 5) -/
def line_equation (x y : ℝ) : Prop := y + 3 = -3 * (x - 5)

/-- The x-intercept of the line -/
def x_intercept : ℝ := 4

/-- The y-intercept of the line -/
def y_intercept : ℝ := 12

/-- The sum of x-intercept and y-intercept is 16 -/
theorem intercept_sum : x_intercept + y_intercept = 16 := by sorry

end intercept_sum_l704_70411


namespace sandys_puppies_l704_70493

/-- Given that Sandy initially had 8 puppies and gave away 4 puppies,
    prove that she now has 4 puppies remaining. -/
theorem sandys_puppies (initial_puppies : ℕ) (puppies_given_away : ℕ)
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = 4) :
  initial_puppies - puppies_given_away = 4 := by
sorry

end sandys_puppies_l704_70493


namespace walking_rate_ratio_l704_70455

theorem walking_rate_ratio (usual_time new_time usual_rate new_rate : ℝ) 
  (h1 : usual_time = 49)
  (h2 : new_time = usual_time - 7)
  (h3 : usual_rate * usual_time = new_rate * new_time) :
  new_rate / usual_rate = 7 / 6 := by
sorry

end walking_rate_ratio_l704_70455


namespace y_squared_mod_30_l704_70466

theorem y_squared_mod_30 (y : ℤ) (h1 : 6 * y ≡ 12 [ZMOD 30]) (h2 : 5 * y ≡ 25 [ZMOD 30]) :
  y^2 ≡ 19 [ZMOD 30] := by
  sorry

end y_squared_mod_30_l704_70466


namespace fifth_term_of_geometric_sequence_l704_70405

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1/2 →
  a 2 * a 4 = 4 * (a 3 - 1) →
  a 5 = 8 := by
  sorry

end fifth_term_of_geometric_sequence_l704_70405


namespace not_perfect_square_l704_70409

theorem not_perfect_square (n : ℕ) : ¬ ∃ m : ℤ, (3^n : ℤ) + 2 * (17^n : ℤ) = m^2 := by
  sorry

end not_perfect_square_l704_70409


namespace optimal_circle_radii_equilateral_triangle_l704_70485

/-- Given an equilateral triangle with side length 1, this theorem states that
    the maximum area covered by three circles centered at the vertices,
    not intersecting each other or the opposite sides, is achieved when
    the radii are R_a = √3/2 and R_b = R_c = 1 - √3/2. -/
theorem optimal_circle_radii_equilateral_triangle :
  let side_length : ℝ := 1
  let height : ℝ := Real.sqrt 3 / 2
  let R_a : ℝ := height
  let R_b : ℝ := 1 - height
  let R_c : ℝ := 1 - height
  let area_covered (r_a r_b r_c : ℝ) : ℝ := π / 6 * (r_a^2 + r_b^2 + r_c^2)
  let is_valid_radii (r_a r_b r_c : ℝ) : Prop :=
    r_a ≤ height ∧ r_a ≥ 1/2 ∧
    r_b ≤ 1 - r_a ∧ r_c ≤ 1 - r_a
  ∀ r_a r_b r_c : ℝ,
    is_valid_radii r_a r_b r_c →
    area_covered r_a r_b r_c ≤ area_covered R_a R_b R_c :=
by sorry

end optimal_circle_radii_equilateral_triangle_l704_70485


namespace ac_length_l704_70450

/-- Given a line segment AB of length 4 with a point C on AB, 
    prove that if AC is the mean proportional between AB and BC, 
    then the length of AC is 2√5 - 2 -/
theorem ac_length (A B C : ℝ) (h1 : B - A = 4) (h2 : A ≤ C ∧ C ≤ B) 
  (h3 : (C - A)^2 = (B - A) * (B - C)) : C - A = 2 * Real.sqrt 5 - 2 := by
  sorry

end ac_length_l704_70450


namespace buses_needed_for_trip_l704_70478

/-- Calculates the number of buses needed for a school trip -/
theorem buses_needed_for_trip (total_students : ℕ) (van_students : ℕ) (bus_capacity : ℕ) 
  (h1 : total_students = 500)
  (h2 : van_students = 56)
  (h3 : bus_capacity = 45) :
  Nat.ceil ((total_students - van_students) / bus_capacity) = 10 := by
  sorry

#check buses_needed_for_trip

end buses_needed_for_trip_l704_70478


namespace total_triangles_in_4_layer_grid_l704_70469

/-- Represents a triangular grid with a given number of layers -/
def TriangularGrid (layers : ℕ) : Type := Unit

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid 4) : ℕ := 10

/-- Counts the number of medium triangles (made of 4 small triangles) in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid 4) : ℕ := 6

/-- Counts the number of large triangles (made of 9 small triangles) in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid 4) : ℕ := 1

/-- The total number of triangles in a 4-layer triangular grid is 17 -/
theorem total_triangles_in_4_layer_grid (grid : TriangularGrid 4) :
  countSmallTriangles grid + countMediumTriangles grid + countLargeTriangles grid = 17 := by
  sorry

end total_triangles_in_4_layer_grid_l704_70469


namespace matrix_power_equality_l704_70458

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 2, 1]

def B : Matrix (Fin 2) (Fin 2) ℕ := !![17, 12; 24, 17]

theorem matrix_power_equality :
  A^10 = B^5 := by sorry

end matrix_power_equality_l704_70458


namespace sum_of_coefficients_l704_70427

/-- The sum of the coefficients of the expanded expression -(2x - 5)(4x + 3(2x - 5)) is -15 -/
theorem sum_of_coefficients : ∃ a b c : ℚ,
  -(2 * X - 5) * (4 * X + 3 * (2 * X - 5)) = a * X^2 + b * X + c ∧ a + b + c = -15 :=
by sorry

end sum_of_coefficients_l704_70427


namespace sqrt_inequality_l704_70414

theorem sqrt_inequality (a : ℝ) (h : a > 5) :
  Real.sqrt (a - 5) - Real.sqrt (a - 3) < Real.sqrt (a - 2) - Real.sqrt a :=
by sorry

end sqrt_inequality_l704_70414


namespace regular_polygon_area_l704_70406

theorem regular_polygon_area (n : ℕ) (R : ℝ) (h : R > 0) :
  (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 3 * R^2 → n = 12 := by
  sorry

end regular_polygon_area_l704_70406


namespace sin_25pi_div_6_l704_70428

theorem sin_25pi_div_6 : Real.sin (25 * π / 6) = 1 / 2 := by
  sorry

end sin_25pi_div_6_l704_70428


namespace triangle_construction_l704_70420

/-- Given a point A, a plane S, and distances ρ, ρₐ, and b-c,
    we can construct a triangle ABC with specific properties. -/
theorem triangle_construction (A : ℝ × ℝ) (S : ℝ × ℝ) (ρ ρₐ : ℝ) (b_minus_c : ℝ) 
  (s a b c : ℝ) :
  -- Side a lies in plane S (represented by the condition that a is real)
  -- One vertex is A (implicit in the construction)
  -- ρ is the inradius
  -- ρₐ is the exradius opposite to side a
  (s = (a + b + c) / 2) →  -- Definition of semiperimeter
  (ρ > 0) →  -- Inradius is positive
  (ρₐ > 0) →  -- Exradius is positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- Triangle sides are positive
  (b - c = b_minus_c) →  -- Given difference of sides
  -- Then the following relationships hold:
  ((s - b) * (s - c) = ρ * ρₐ) ∧
  ((s - c) - (s - b) = b - c) ∧
  (Real.sqrt ((s - b) * (s - c)) = Real.sqrt (ρ * ρₐ)) :=
by sorry

end triangle_construction_l704_70420


namespace animal_population_l704_70401

theorem animal_population (lions leopards elephants : ℕ) : 
  lions = 200 →
  lions = 2 * leopards →
  elephants = (lions + leopards) / 2 →
  lions + leopards + elephants = 450 := by
sorry

end animal_population_l704_70401


namespace greatest_distance_between_circle_centers_l704_70403

/-- The greatest possible distance between the centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 15)
  (h_height : rectangle_height = 10)
  (h_diameter : circle_diameter = 5)
  (h_nonneg_width : 0 ≤ rectangle_width)
  (h_nonneg_height : 0 ≤ rectangle_height)
  (h_nonneg_diameter : 0 ≤ circle_diameter)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 5 * Real.sqrt 5 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end greatest_distance_between_circle_centers_l704_70403


namespace max_value_theorem_l704_70456

theorem max_value_theorem (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h : (a₂ - a₁)^2 + (a₃ - a₂)^2 + (a₄ - a₃)^2 + (a₅ - a₄)^2 + (a₆ - a₅)^2 = 1) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ 
  ∀ (x : ℝ), x = (a₅ + a₆) - (a₁ + a₄) → x ≤ M :=
by sorry

end max_value_theorem_l704_70456


namespace johns_tour_program_l704_70489

theorem johns_tour_program (total_budget : ℕ) (budget_reduction : ℕ) (extra_days : ℕ) :
  total_budget = 360 ∧ budget_reduction = 3 ∧ extra_days = 4 →
  ∃ (days : ℕ) (daily_expense : ℕ),
    total_budget = days * daily_expense ∧
    total_budget = (days + extra_days) * (daily_expense - budget_reduction) ∧
    days = 20 := by
  sorry

end johns_tour_program_l704_70489


namespace complex_modulus_equality_l704_70483

theorem complex_modulus_equality (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end complex_modulus_equality_l704_70483


namespace square_area_ratio_l704_70492

theorem square_area_ratio (big_side : ℝ) (small_side : ℝ) 
  (h1 : big_side = 12)
  (h2 : small_side = 6) : 
  (small_side ^ 2) / (big_side ^ 2 - small_side ^ 2) = 1 / 3 := by
  sorry

end square_area_ratio_l704_70492


namespace solution_of_system_l704_70441

theorem solution_of_system (x y : ℚ) 
  (eq1 : 3 * y - 4 * x = 8)
  (eq2 : 2 * y + x = -1) : 
  x = -19/11 ∧ y = 4/11 := by
sorry

end solution_of_system_l704_70441


namespace can_distribution_properties_l704_70438

/-- Represents the distribution of cans across bags -/
structure CanDistribution where
  total_cans : ℕ
  num_bags : ℕ
  first_bags_limit : ℕ
  last_bags_limit : ℕ

/-- Calculates the number of cans in each of the last bags -/
def cans_in_last_bags (d : CanDistribution) : ℕ :=
  let cans_in_first_bags := d.first_bags_limit * (d.num_bags / 2)
  let remaining_cans := d.total_cans - cans_in_first_bags
  remaining_cans / (d.num_bags / 2)

/-- Calculates the difference between cans in first and last bag -/
def cans_difference (d : CanDistribution) : ℕ :=
  d.first_bags_limit - cans_in_last_bags d

/-- Theorem stating the properties of the can distribution -/
theorem can_distribution_properties (d : CanDistribution) 
    (h1 : d.total_cans = 200)
    (h2 : d.num_bags = 6)
    (h3 : d.first_bags_limit = 40)
    (h4 : d.last_bags_limit = 30) :
    cans_in_last_bags d = 26 ∧ cans_difference d = 14 := by
  sorry

#eval cans_in_last_bags { total_cans := 200, num_bags := 6, first_bags_limit := 40, last_bags_limit := 30 }
#eval cans_difference { total_cans := 200, num_bags := 6, first_bags_limit := 40, last_bags_limit := 30 }

end can_distribution_properties_l704_70438


namespace train_length_l704_70421

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 27 → ∃ (length_m : ℝ), abs (length_m - 450.09) < 0.01 := by
  sorry

#check train_length

end train_length_l704_70421


namespace average_monthly_increase_l704_70467

/-- Represents the monthly growth rate as a real number between 0 and 1 -/
def monthly_growth_rate : ℝ := sorry

/-- The initial turnover in January in millions of yuan -/
def initial_turnover : ℝ := 2

/-- The turnover in March in millions of yuan -/
def march_turnover : ℝ := 2.88

/-- The number of months between January and March -/
def months_passed : ℕ := 2

theorem average_monthly_increase :
  initial_turnover * (1 + monthly_growth_rate) ^ months_passed = march_turnover ∧
  monthly_growth_rate = 0.2 := by sorry

end average_monthly_increase_l704_70467


namespace solve_system_l704_70418

theorem solve_system (u v : ℝ) 
  (eq1 : 3 * u - 7 * v = 29)
  (eq2 : 5 * u + 3 * v = -9) :
  u + v = -3.363 := by sorry

end solve_system_l704_70418


namespace three_men_five_jobs_earnings_l704_70445

/-- Calculates the total earnings for a group of workers completing multiple jobs -/
def totalEarnings (numWorkers : ℕ) (numJobs : ℕ) (hourlyRate : ℕ) (hoursPerJob : ℕ) : ℕ :=
  numWorkers * numJobs * hourlyRate * hoursPerJob

/-- Proves that 3 men working on 5 jobs at $10 per hour, with each job taking 1 hour, earn $150 in total -/
theorem three_men_five_jobs_earnings :
  totalEarnings 3 5 10 1 = 150 := by
  sorry

end three_men_five_jobs_earnings_l704_70445


namespace work_left_after_collaboration_l704_70433

/-- Represents the fraction of work left after two workers collaborate for a given number of days. -/
def work_left (a_days b_days collab_days : ℕ) : ℚ :=
  1 - (collab_days : ℚ) * (1 / a_days + 1 / b_days)

/-- Theorem stating that if A can complete the work in 15 days and B in 20 days,
    then after working together for 4 days, 8/15 of the work is left. -/
theorem work_left_after_collaboration :
  work_left 15 20 4 = 8/15 := by
  sorry

end work_left_after_collaboration_l704_70433


namespace local_minimum_implies_b_range_l704_70452

def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

theorem local_minimum_implies_b_range (b : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 1, IsLocalMin (f b) x₀) →
  0 < b ∧ b < 1 := by
sorry

end local_minimum_implies_b_range_l704_70452


namespace min_value_expression_equality_condition_l704_70446

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) = 3 ↔ y = x * Real.sqrt 2 :=
by sorry

end min_value_expression_equality_condition_l704_70446


namespace oak_trees_in_park_l704_70426

theorem oak_trees_in_park (current_trees : ℕ) 
  (h1 : current_trees + 4 = 9) : current_trees = 5 := by
  sorry

end oak_trees_in_park_l704_70426


namespace inequality_system_solution_l704_70476

theorem inequality_system_solution (m : ℝ) : 
  (∀ x, (x + 5 < 4*x - 1 ∧ x > m) ↔ x > 2) → m ≤ 2 := by
  sorry

end inequality_system_solution_l704_70476


namespace double_square_root_simplification_l704_70471

theorem double_square_root_simplification (a b m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 2 * Real.sqrt b > 0) 
  (hm : m > 0) (hn : n > 0)
  (h1 : Real.sqrt m ^ 2 + Real.sqrt n ^ 2 = a)
  (h2 : Real.sqrt m * Real.sqrt n = Real.sqrt b) :
  Real.sqrt (a + 2 * Real.sqrt b) = |Real.sqrt m + Real.sqrt n| ∧
  Real.sqrt (a - 2 * Real.sqrt b) = |Real.sqrt m - Real.sqrt n| :=
by sorry

end double_square_root_simplification_l704_70471


namespace tangent_slope_at_point_one_l704_70472

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_point_one :
  f' 1 = 1 :=
sorry

end tangent_slope_at_point_one_l704_70472


namespace sandcastle_height_difference_l704_70496

/-- The height difference between Miki's sandcastle and her sister's sandcastle -/
theorem sandcastle_height_difference 
  (miki_height : ℝ) 
  (sister_height : ℝ) 
  (h1 : miki_height = 0.8333333333333334) 
  (h2 : sister_height = 0.5) : 
  miki_height - sister_height = 0.3333333333333334 := by
  sorry

end sandcastle_height_difference_l704_70496


namespace tip_amount_is_36_dollars_l704_70448

/-- The cost of a woman's haircut in dollars -/
def womens_haircut_cost : ℝ := 48

/-- The cost of a child's haircut in dollars -/
def childs_haircut_cost : ℝ := 36

/-- The cost of a teenager's haircut in dollars -/
def teens_haircut_cost : ℝ := 40

/-- The cost of Tayzia's hair treatment in dollars -/
def hair_treatment_cost : ℝ := 20

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.20

/-- The total cost of haircuts and treatment before tip -/
def total_cost : ℝ :=
  womens_haircut_cost + 2 * childs_haircut_cost + teens_haircut_cost + hair_treatment_cost

/-- The theorem stating that the 20% tip is $36 -/
theorem tip_amount_is_36_dollars : tip_percentage * total_cost = 36 := by
  sorry

end tip_amount_is_36_dollars_l704_70448


namespace lowest_cost_option_c_l704_70434

/-- Represents a shipping option with a flat fee and per-pound rate -/
structure ShippingOption where
  flatFee : ℝ
  perPoundRate : ℝ

/-- Calculates the total cost for a given shipping option and weight -/
def totalCost (option : ShippingOption) (weight : ℝ) : ℝ :=
  option.flatFee + option.perPoundRate * weight

/-- The three shipping options available -/
def optionA : ShippingOption := ⟨5.00, 0.80⟩
def optionB : ShippingOption := ⟨4.50, 0.85⟩
def optionC : ShippingOption := ⟨3.00, 0.95⟩

/-- The weight of the package in pounds -/
def packageWeight : ℝ := 5

theorem lowest_cost_option_c :
  let costA := totalCost optionA packageWeight
  let costB := totalCost optionB packageWeight
  let costC := totalCost optionC packageWeight
  (costC < costA ∧ costC < costB) ∧ costC = 7.75 := by
  sorry

end lowest_cost_option_c_l704_70434


namespace circle_sum_l704_70494

theorem circle_sum (square circle : ℚ) 
  (eq1 : 2 * square + 3 * circle = 26)
  (eq2 : 3 * square + 2 * circle = 23) :
  4 * circle = 128 / 5 := by
sorry

end circle_sum_l704_70494


namespace zero_point_existence_l704_70430

def f (x : ℝ) := x^3 + 2*x - 5

theorem zero_point_existence :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : f 1 < 0 := sorry
  have h3 : f 2 > 0 := sorry
  sorry

end zero_point_existence_l704_70430
