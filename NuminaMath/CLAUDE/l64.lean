import Mathlib

namespace NUMINAMATH_CALUDE_alpha_not_rational_l64_6434

theorem alpha_not_rational (α : ℝ) (h : Real.cos (α * π / 180) = 1/3) : ¬ (∃ (m n : ℤ), α = m / n) := by
  sorry

end NUMINAMATH_CALUDE_alpha_not_rational_l64_6434


namespace NUMINAMATH_CALUDE_profit_maximization_l64_6453

/-- Profit function given price x -/
def profit (x : ℝ) : ℝ := (x - 40) * (300 - (x - 60) * 10)

/-- The price that maximizes profit -/
def optimal_price : ℝ := 65

/-- The maximum profit achieved -/
def max_profit : ℝ := 6250

theorem profit_maximization :
  (∀ x : ℝ, profit x ≤ profit optimal_price) ∧
  profit optimal_price = max_profit := by
  sorry

#check profit_maximization

end NUMINAMATH_CALUDE_profit_maximization_l64_6453


namespace NUMINAMATH_CALUDE_x_value_when_y_is_one_l64_6404

theorem x_value_when_y_is_one (x y : ℝ) : 
  y = 2 / (4 * x + 2) → y = 1 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_one_l64_6404


namespace NUMINAMATH_CALUDE_equation_solution_l64_6405

theorem equation_solution :
  ∃ x : ℝ, (2*x + 1)/3 - (5*x - 1)/6 = 1 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l64_6405


namespace NUMINAMATH_CALUDE_hotel_room_pricing_l64_6483

theorem hotel_room_pricing (total_rooms : ℕ) (double_rooms : ℕ) (single_room_cost : ℕ) (total_revenue : ℕ) :
  total_rooms = 260 →
  double_rooms = 196 →
  single_room_cost = 35 →
  total_revenue = 14000 →
  ∃ (double_room_cost : ℕ),
    double_room_cost = 60 ∧
    total_revenue = (total_rooms - double_rooms) * single_room_cost + double_rooms * double_room_cost :=
by
  sorry

#check hotel_room_pricing

end NUMINAMATH_CALUDE_hotel_room_pricing_l64_6483


namespace NUMINAMATH_CALUDE_restaurant_menu_theorem_l64_6470

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem restaurant_menu_theorem (v : ℕ) : 
  (choose 5 2 * choose v 2 > 200) → v ≥ 7 := by sorry

end NUMINAMATH_CALUDE_restaurant_menu_theorem_l64_6470


namespace NUMINAMATH_CALUDE_num_cubes_5_peaks_num_cubes_2014_peaks_painted_area_2014_peaks_l64_6461

/-- Represents a wall made of unit cubes with a given number of peaks -/
structure Wall where
  peaks : ℕ

/-- The number of cubes needed to construct a wall with n peaks -/
def num_cubes (w : Wall) : ℕ := 3 * w.peaks - 1

/-- The painted surface area of a wall with n peaks, excluding the base -/
def painted_area (w : Wall) : ℕ := 10 * w.peaks + 9

/-- Theorem stating the number of cubes for a wall with 5 peaks -/
theorem num_cubes_5_peaks : num_cubes { peaks := 5 } = 14 := by sorry

/-- Theorem stating the number of cubes for a wall with 2014 peaks -/
theorem num_cubes_2014_peaks : num_cubes { peaks := 2014 } = 6041 := by sorry

/-- Theorem stating the painted area for a wall with 2014 peaks -/
theorem painted_area_2014_peaks : painted_area { peaks := 2014 } = 20139 := by sorry

end NUMINAMATH_CALUDE_num_cubes_5_peaks_num_cubes_2014_peaks_painted_area_2014_peaks_l64_6461


namespace NUMINAMATH_CALUDE_abs_negative_2023_l64_6437

theorem abs_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l64_6437


namespace NUMINAMATH_CALUDE_product_equals_zero_l64_6403

theorem product_equals_zero : (3 * 5 * 7 + 4 * 6 * 8) * (2 * 12 * 5 - 20 * 3 * 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l64_6403


namespace NUMINAMATH_CALUDE_total_books_count_l64_6450

/-- The number of books Susan has -/
def susan_books : ℕ := 600

/-- The number of books Lidia has -/
def lidia_books : ℕ := 4 * susan_books

/-- The total number of books Susan and Lidia have -/
def total_books : ℕ := susan_books + lidia_books

theorem total_books_count : total_books = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l64_6450


namespace NUMINAMATH_CALUDE_star_cell_is_one_l64_6448

/-- Represents a 4x4 grid of natural numbers -/
def Grid := Fin 4 → Fin 4 → Nat

/-- Check if all numbers in the grid are nonzero -/
def all_nonzero (g : Grid) : Prop :=
  ∀ i j, g i j ≠ 0

/-- Calculate the product of a row -/
def row_product (g : Grid) (i : Fin 4) : Nat :=
  (g i 0) * (g i 1) * (g i 2) * (g i 3)

/-- Calculate the product of a column -/
def col_product (g : Grid) (j : Fin 4) : Nat :=
  (g 0 j) * (g 1 j) * (g 2 j) * (g 3 j)

/-- Calculate the product of the main diagonal -/
def main_diag_product (g : Grid) : Nat :=
  (g 0 0) * (g 1 1) * (g 2 2) * (g 3 3)

/-- Calculate the product of the anti-diagonal -/
def anti_diag_product (g : Grid) : Nat :=
  (g 0 3) * (g 1 2) * (g 2 1) * (g 3 0)

/-- Check if all products are equal -/
def all_products_equal (g : Grid) : Prop :=
  let p := row_product g 0
  (∀ i, row_product g i = p) ∧
  (∀ j, col_product g j = p) ∧
  (main_diag_product g = p) ∧
  (anti_diag_product g = p)

/-- The main theorem -/
theorem star_cell_is_one (g : Grid) 
  (h1 : all_nonzero g)
  (h2 : all_products_equal g)
  (h3 : g 1 1 = 2)
  (h4 : g 1 2 = 16)
  (h5 : g 2 1 = 8)
  (h6 : g 2 2 = 32) :
  g 1 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_star_cell_is_one_l64_6448


namespace NUMINAMATH_CALUDE_vector_b_coordinates_l64_6409

theorem vector_b_coordinates (a b : ℝ × ℝ) :
  a = (Real.sqrt 3, Real.sqrt 5) →
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (b.1^2 + b.2^2 = 4) →
  (b = (-Real.sqrt 10 / 2, Real.sqrt 6 / 2) ∨ b = (Real.sqrt 10 / 2, -Real.sqrt 6 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_coordinates_l64_6409


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l64_6459

def anna_lap_time : ℕ := 5
def stephanie_lap_time : ℕ := 8
def james_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [anna_lap_time, stephanie_lap_time, james_lap_time]
  Nat.lcm (Nat.lcm anna_lap_time stephanie_lap_time) james_lap_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l64_6459


namespace NUMINAMATH_CALUDE_complement_union_theorem_l64_6414

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l64_6414


namespace NUMINAMATH_CALUDE_union_M_N_complement_N_P_subset_M_iff_l64_6401

-- Define the sets M, N, and P
def M : Set ℝ := {x | (x + 4) * (x - 6) < 0}
def N : Set ℝ := {x | x - 5 < 0}
def P (t : ℝ) : Set ℝ := {x | |x| = t}

-- Theorem 1: M ∪ N = {x | x < 6}
theorem union_M_N : M ∪ N = {x | x < 6} := by sorry

-- Theorem 2: N̄ₘ = {x | x ≥ 5}
theorem complement_N : (Nᶜ : Set ℝ) = {x | x ≥ 5} := by sorry

-- Theorem 3: P ⊆ M if and only if t ∈ (-∞, 4)
theorem P_subset_M_iff (t : ℝ) : P t ⊆ M ↔ t < 4 := by sorry

end NUMINAMATH_CALUDE_union_M_N_complement_N_P_subset_M_iff_l64_6401


namespace NUMINAMATH_CALUDE_smallest_q_for_five_in_range_l64_6431

/-- The function g(x) defined as x^2 - 4x + q -/
def g (q : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + q

/-- 5 is within the range of g(x) -/
def in_range (q : ℝ) : Prop := ∃ x, g q x = 5

/-- The smallest value of q such that 5 is within the range of g(x) is 9 -/
theorem smallest_q_for_five_in_range : 
  (∃ q₀, in_range q₀ ∧ ∀ q, in_range q → q₀ ≤ q) ∧ 
  (∀ q, in_range q ↔ 9 ≤ q) :=
sorry

end NUMINAMATH_CALUDE_smallest_q_for_five_in_range_l64_6431


namespace NUMINAMATH_CALUDE_helen_lawn_gas_consumption_l64_6422

/-- Represents the number of months with 2 cuts per month -/
def low_frequency_months : ℕ := 4

/-- Represents the number of months with 4 cuts per month -/
def high_frequency_months : ℕ := 4

/-- Represents the number of cuts per month in low frequency months -/
def low_frequency_cuts : ℕ := 2

/-- Represents the number of cuts per month in high frequency months -/
def high_frequency_cuts : ℕ := 4

/-- Represents the number of cuts before needing to refuel -/
def cuts_per_refuel : ℕ := 4

/-- Represents the number of gallons used per refuel -/
def gallons_per_refuel : ℕ := 2

/-- Theorem stating that Helen will need 12 gallons of gas for lawn cutting from March through October -/
theorem helen_lawn_gas_consumption : 
  (low_frequency_months * low_frequency_cuts + high_frequency_months * high_frequency_cuts) / cuts_per_refuel * gallons_per_refuel = 12 :=
by sorry

end NUMINAMATH_CALUDE_helen_lawn_gas_consumption_l64_6422


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l64_6433

/-- The cost of chocolate bars for a scout camp -/
theorem chocolate_bar_cost (chocolate_bar_cost : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : 
  chocolate_bar_cost = 1.5 →
  sections_per_bar = 3 →
  num_scouts = 15 →
  smores_per_scout = 2 →
  (num_scouts * smores_per_scout : ℝ) / sections_per_bar * chocolate_bar_cost = 15 := by
  sorry

#check chocolate_bar_cost

end NUMINAMATH_CALUDE_chocolate_bar_cost_l64_6433


namespace NUMINAMATH_CALUDE_fraction_of_5000_l64_6464

theorem fraction_of_5000 : 
  ∃ (f : ℚ), (f * (1/2 * (2/5 * 5000)) = 750.0000000000001) ∧ (f = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_5000_l64_6464


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l64_6443

theorem polygon_sides_from_angle_sum (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 900 → (n - 2) * 180 = sum_angles → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l64_6443


namespace NUMINAMATH_CALUDE_train_crossing_time_l64_6417

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 56 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l64_6417


namespace NUMINAMATH_CALUDE_polyline_segment_bound_l64_6408

/-- Represents a grid paper with square side length 1 -/
structure GridPaper where
  -- Additional structure properties can be added if needed

/-- Represents a point on the grid paper -/
structure GridPoint where
  -- Additional point properties can be added if needed

/-- Represents a polyline segment on the grid paper -/
structure PolylineSegment where
  start : GridPoint
  length : ℕ
  -- Additional segment properties can be added if needed

/-- 
  P_k denotes the number of different polyline segments of length k 
  starting from a fixed point O on a grid paper, where each segment 
  lies along the grid lines
-/
def P (grid : GridPaper) (O : GridPoint) (k : ℕ) : ℕ :=
  sorry -- Definition of P_k

/-- 
  Theorem: For all natural numbers k, the number of different polyline 
  segments of length k starting from a fixed point O on a grid paper 
  with square side length 1, where each segment lies along the grid lines, 
  is less than 2 × 3^k
-/
theorem polyline_segment_bound 
  (grid : GridPaper) (O : GridPoint) : 
  ∀ k : ℕ, P grid O k < 2 * 3^k := by
  sorry


end NUMINAMATH_CALUDE_polyline_segment_bound_l64_6408


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l64_6449

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 40*x^2 + 400 = 0 → x ≥ -2*Real.sqrt 5 ∧ (∃ y, y^4 - 40*y^2 + 400 = 0 ∧ y = -2*Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l64_6449


namespace NUMINAMATH_CALUDE_average_weight_of_group_l64_6444

theorem average_weight_of_group (girls_count boys_count : ℕ) 
  (girls_avg_weight boys_avg_weight : ℝ) :
  girls_count = 5 →
  boys_count = 5 →
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  let total_count := girls_count + boys_count
  let total_weight := girls_count * girls_avg_weight + boys_count * boys_avg_weight
  (total_weight / total_count : ℝ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_group_l64_6444


namespace NUMINAMATH_CALUDE_sum_coefficients_expansion_l64_6427

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the sum of coefficients function
def sumCoefficients (x : ℕ) : ℕ :=
  (C x 1 + C (x+1) 1 + C (x+2) 1 + C (x+3) 1) ^ 2

-- Theorem statement
theorem sum_coefficients_expansion :
  ∃ x : ℕ, sumCoefficients x = 225 :=
sorry

end NUMINAMATH_CALUDE_sum_coefficients_expansion_l64_6427


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l64_6487

theorem cubic_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 0) : 
  (x^3 + y^3 + z^3) / (x*y*z) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l64_6487


namespace NUMINAMATH_CALUDE_lung_cancer_probability_l64_6420

theorem lung_cancer_probability (overall_prob : ℝ) (smoker_ratio : ℝ) (smoker_prob : ℝ) :
  overall_prob = 0.001 →
  smoker_ratio = 0.2 →
  smoker_prob = 0.004 →
  ∃ (nonsmoker_prob : ℝ),
    nonsmoker_prob = 0.00025 ∧
    overall_prob = smoker_ratio * smoker_prob + (1 - smoker_ratio) * nonsmoker_prob :=
by sorry

end NUMINAMATH_CALUDE_lung_cancer_probability_l64_6420


namespace NUMINAMATH_CALUDE_initial_money_calculation_l64_6486

theorem initial_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 840)
  (h2 : spent_percentage = 0.3)
  : (remaining_money / (1 - spent_percentage)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l64_6486


namespace NUMINAMATH_CALUDE_no_simultaneous_solution_l64_6423

theorem no_simultaneous_solution : ¬∃ x : ℝ, (5 * x^2 - 7 * x + 1 < 0) ∧ (x^2 - 9 * x + 30 < 0) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_solution_l64_6423


namespace NUMINAMATH_CALUDE_martin_distance_l64_6498

/-- The distance traveled by Martin -/
def distance : ℝ := 72.0

/-- Martin's driving speed in miles per hour -/
def speed : ℝ := 12.0

/-- Time taken for Martin's journey in hours -/
def time : ℝ := 6.0

/-- Theorem stating that the distance Martin traveled is equal to his speed multiplied by the time taken -/
theorem martin_distance : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_martin_distance_l64_6498


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l64_6465

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l64_6465


namespace NUMINAMATH_CALUDE_power_inequality_l64_6419

theorem power_inequality (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l64_6419


namespace NUMINAMATH_CALUDE_function_value_range_l64_6412

theorem function_value_range (a : ℝ) : 
  (∃ x y : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ y ∈ Set.Icc (-1 : ℝ) 1 ∧ 
   (a * x + 2 * a + 1) * (a * y + 2 * a + 1) < 0) ↔ 
  a ∈ Set.Ioo (-(1/3) : ℝ) (-1) :=
by sorry

end NUMINAMATH_CALUDE_function_value_range_l64_6412


namespace NUMINAMATH_CALUDE_frustum_smaller_radius_l64_6436

/-- A circular frustum with the given properties -/
structure CircularFrustum where
  r : ℝ  -- radius of the smaller base
  slant_height : ℝ
  lateral_area : ℝ

/-- The theorem statement -/
theorem frustum_smaller_radius (f : CircularFrustum) 
  (h1 : f.slant_height = 3)
  (h2 : f.lateral_area = 84 * Real.pi)
  (h3 : 2 * Real.pi * (3 * f.r) = 3 * (2 * Real.pi * f.r)) :
  f.r = 7 := by
  sorry

end NUMINAMATH_CALUDE_frustum_smaller_radius_l64_6436


namespace NUMINAMATH_CALUDE_corrected_mean_l64_6451

theorem corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ initial_mean = 32 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n * initial_mean - incorrect_value + correct_value) / n = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l64_6451


namespace NUMINAMATH_CALUDE_cone_no_rectangular_cross_section_cone_unique_no_rectangular_cross_section_l64_6406

-- Define the types of geometric shapes we're considering
inductive GeometricShape
| Cone
| Cylinder
| TriangularPrism
| RectangularPrism

-- Define a function that determines if a shape can have a rectangular cross-section
def canHaveRectangularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Cone => False
  | _ => True

-- Theorem statement
theorem cone_no_rectangular_cross_section :
  ∀ (shape : GeometricShape),
    ¬(canHaveRectangularCrossSection shape) ↔ shape = GeometricShape.Cone :=
by
  sorry

-- Alternative formulation focusing on the unique property of the cone
theorem cone_unique_no_rectangular_cross_section :
  ∃! (shape : GeometricShape), ¬(canHaveRectangularCrossSection shape) :=
by
  sorry

end NUMINAMATH_CALUDE_cone_no_rectangular_cross_section_cone_unique_no_rectangular_cross_section_l64_6406


namespace NUMINAMATH_CALUDE_train_crossing_time_l64_6462

theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 100 ∧ train_speed_kmh = 90 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l64_6462


namespace NUMINAMATH_CALUDE_jills_nickels_l64_6447

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

end NUMINAMATH_CALUDE_jills_nickels_l64_6447


namespace NUMINAMATH_CALUDE_complex_number_calculation_l64_6454

theorem complex_number_calculation (z : ℂ) : z = 1 + I → (2 / z) + z^2 = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l64_6454


namespace NUMINAMATH_CALUDE_cups_brought_to_class_l64_6426

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

end NUMINAMATH_CALUDE_cups_brought_to_class_l64_6426


namespace NUMINAMATH_CALUDE_least_possible_beta_l64_6481

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

end NUMINAMATH_CALUDE_least_possible_beta_l64_6481


namespace NUMINAMATH_CALUDE_bird_on_time_speed_l64_6438

/-- Represents the problem of Mr. Bird's commute --/
structure BirdCommute where
  distance : ℝ
  time_on_time : ℝ
  speed_late : ℝ
  speed_early : ℝ
  late_time : ℝ
  early_time : ℝ

/-- The theorem stating the correct speed for Mr. Bird to arrive on time --/
theorem bird_on_time_speed (b : BirdCommute) 
  (h1 : b.speed_late = 30)
  (h2 : b.speed_early = 50)
  (h3 : b.late_time = 5 / 60)
  (h4 : b.early_time = 5 / 60)
  (h5 : b.distance = b.speed_late * (b.time_on_time + b.late_time))
  (h6 : b.distance = b.speed_early * (b.time_on_time - b.early_time)) :
  b.distance / b.time_on_time = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_bird_on_time_speed_l64_6438


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_five_l64_6499

-- Define the initial number of men and women
variable (M W : ℕ)

-- Define the final number of men and women
def final_men := M + 2
def final_women := 2 * (W - 3)

-- Theorem statement
theorem initial_ratio_is_four_to_five : 
  final_men = 14 ∧ final_women = 24 → M * 5 = W * 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_five_l64_6499


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_m_range_l64_6467

theorem quadratic_equation_real_roots_m_range 
  (m : ℝ) 
  (has_real_roots : ∃ x : ℝ, (m - 2) * x^2 + 2 * m * x + m + 3 = 0) :
  m ≤ 6 ∧ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_m_range_l64_6467


namespace NUMINAMATH_CALUDE_even_function_positive_x_l64_6493

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_positive_x 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_neg : ∀ x < 0, f x = x * (x - 1)) : 
  ∀ x > 0, f x = x * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_even_function_positive_x_l64_6493


namespace NUMINAMATH_CALUDE_path_area_is_775_l64_6457

/-- Represents the dimensions of a rectangular field with a surrounding path -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ

/-- Calculates the area of the path surrounding a rectangular field -/
def path_area (f : FieldWithPath) : ℝ :=
  (f.field_length + 2 * f.path_width) * (f.field_width + 2 * f.path_width) -
  f.field_length * f.field_width

/-- Theorem stating that the area of the path for the given field dimensions is 775 sq m -/
theorem path_area_is_775 :
  let f : FieldWithPath := {
    field_length := 95,
    field_width := 55,
    path_width := 2.5
  }
  path_area f = 775 := by sorry

end NUMINAMATH_CALUDE_path_area_is_775_l64_6457


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l64_6475

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

-- State the theorem
theorem complement_of_P_in_U : 
  Set.compl P = Set.Ioo (-1 : ℝ) (6 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l64_6475


namespace NUMINAMATH_CALUDE_partner_A_money_received_l64_6418

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

end NUMINAMATH_CALUDE_partner_A_money_received_l64_6418


namespace NUMINAMATH_CALUDE_special_triangle_ac_length_l64_6452

/-- A triangle ABC with a point D on side AC, satisfying specific conditions -/
structure SpecialTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point D on side AC -/
  D : ℝ × ℝ
  /-- AB is greater than BC -/
  ab_gt_bc : dist A B > dist B C
  /-- BC equals 6 -/
  bc_eq_six : dist B C = 6
  /-- BD equals 7 -/
  bd_eq_seven : dist B D = 7
  /-- Triangle ABD is isosceles -/
  abd_isosceles : dist A B = dist A D ∨ dist A B = dist B D
  /-- Triangle BCD is isosceles -/
  bcd_isosceles : dist B C = dist C D ∨ dist B D = dist C D
  /-- D lies on AC -/
  d_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = ((1 - t) • A.1 + t • C.1, (1 - t) • A.2 + t • C.2)

/-- The length of AC in the special triangle is 13 -/
theorem special_triangle_ac_length (t : SpecialTriangle) : dist t.A t.C = 13 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_ac_length_l64_6452


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l64_6413

theorem consecutive_integers_square_sum (n : ℕ) : 
  (n > 0) → 
  (n^2 + (n + 1)^2 = n * (n + 1) + 91) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l64_6413


namespace NUMINAMATH_CALUDE_orange_bin_count_l64_6480

theorem orange_bin_count (initial : ℕ) (removed : ℕ) (added : ℕ) :
  initial = 40 →
  removed = 37 →
  added = 7 →
  initial - removed + added = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_count_l64_6480


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_remainder_is_z_minus_one_l64_6435

/-- The polynomial division theorem for this specific case -/
theorem polynomial_division_theorem (z : ℂ) :
  ∃ (Q R : ℂ → ℂ), z^2023 + 1 = (z^2 - z + 1) * Q z + R z ∧ 
  (∀ x, ∃ (a b : ℂ), R x = a * x + b) := by sorry

/-- The main theorem proving R(z) = z - 1 -/
theorem remainder_is_z_minus_one :
  ∃ (Q R : ℂ → ℂ), 
    (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) ∧
    (∀ x, ∃ (a b : ℂ), R x = a * x + b) ∧
    (∀ z, R z = z - 1) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_remainder_is_z_minus_one_l64_6435


namespace NUMINAMATH_CALUDE_num_pyramids_eq_106_l64_6425

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

end NUMINAMATH_CALUDE_num_pyramids_eq_106_l64_6425


namespace NUMINAMATH_CALUDE_prop_A_prop_B_prop_C_false_prop_D_main_theorem_l64_6485

-- Define the basic concepts
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry

-- Define the relationships
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def intersect (l : Line) (p : Plane) : Prop := sorry
def onPlane (p : Point) (pl : Plane) : Prop := sorry
def angle (l1 l2 : Line) : ℝ := sorry
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Proposition A
theorem prop_A (l1 l2 l3 : Line) :
  parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2 := sorry

-- Proposition B
theorem prop_B (a b c : Line) (θ : ℝ) :
  parallel a b → ¬(intersect c a) → ¬(intersect c b) → angle c a = θ → angle c b = θ := sorry

-- Proposition C (false statement)
theorem prop_C_false :
  ∃ (p1 p2 p3 p4 : Point) (pl : Plane), 
    ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl) →
    ¬(collinear p1 p2 p3 ∨ collinear p1 p2 p4 ∨ collinear p1 p3 p4 ∨ collinear p2 p3 p4) := sorry

-- Proposition D
theorem prop_D (a : Line) (α : Plane) (P : Point) :
  parallel a α → onPlane P α → ∃ (l : Line), parallel l a ∧ onPlane P l ∧ (∀ (Q : Point), onPlane Q l → onPlane Q α) := sorry

-- Main theorem stating that A, B, and D are true while C is false
theorem main_theorem : 
  (∀ l1 l2 l3, parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2) ∧
  (∀ a b c θ, parallel a b → ¬(intersect c a) → ¬(intersect c b) → angle c a = θ → angle c b = θ) ∧
  (∃ p1 p2 p3 p4 pl, ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl) ∧
    (collinear p1 p2 p3 ∨ collinear p1 p2 p4 ∨ collinear p1 p3 p4 ∨ collinear p2 p3 p4)) ∧
  (∀ a α P, parallel a α → onPlane P α → 
    ∃ l, parallel l a ∧ onPlane P l ∧ (∀ Q, onPlane Q l → onPlane Q α)) := sorry

end NUMINAMATH_CALUDE_prop_A_prop_B_prop_C_false_prop_D_main_theorem_l64_6485


namespace NUMINAMATH_CALUDE_point_division_ratios_l64_6407

/-- Given two points A and B on a line, there exist points M and N such that
    AM:MB = 2:1 and AN:NB = 1:3 respectively. -/
theorem point_division_ratios (A B : ℝ) : 
  (∃ M : ℝ, |A - M| / |M - B| = 2) ∧ 
  (∃ N : ℝ, |A - N| / |N - B| = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_point_division_ratios_l64_6407


namespace NUMINAMATH_CALUDE_mark_additional_spending_l64_6442

def mark_spending (initial_amount : ℝ) (additional_first_store : ℝ) : Prop :=
  let half_spent := initial_amount / 2
  let remaining_after_half := initial_amount - half_spent
  let remaining_after_first := remaining_after_half - additional_first_store
  let third_spent := initial_amount / 3
  let remaining_after_second := remaining_after_first - third_spent - 16
  remaining_after_second = 0

theorem mark_additional_spending :
  mark_spending 180 14 := by sorry

end NUMINAMATH_CALUDE_mark_additional_spending_l64_6442


namespace NUMINAMATH_CALUDE_monotonicity_and_range_l64_6497

noncomputable def f (a x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + a^2

theorem monotonicity_and_range :
  (∀ x > 0, ∀ y > 0, (2-Real.sqrt 2)/2 < x → x < y → y < (2+Real.sqrt 2)/2 → f 2 y < f 2 x) ∧
  (∀ x > 0, ∀ y > 0, 0 < x → x < y → y < (2-Real.sqrt 2)/2 → f 2 x < f 2 y) ∧
  (∀ x > 0, ∀ y > 0, (2+Real.sqrt 2)/2 < x → x < y → f 2 x < f 2 y) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, x < y → f a x ≥ f a y) → a ≥ 19/6) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_l64_6497


namespace NUMINAMATH_CALUDE_negation_of_forall_leq_is_exists_gt_l64_6410

theorem negation_of_forall_leq_is_exists_gt (p : (n : ℕ) → n^2 ≤ 2^n → Prop) :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_leq_is_exists_gt_l64_6410


namespace NUMINAMATH_CALUDE_mothers_carrots_l64_6421

theorem mothers_carrots (faye_carrots good_carrots bad_carrots : ℕ) 
  (h1 : faye_carrots = 23)
  (h2 : good_carrots = 12)
  (h3 : bad_carrots = 16) :
  good_carrots + bad_carrots - faye_carrots = 5 :=
by sorry

end NUMINAMATH_CALUDE_mothers_carrots_l64_6421


namespace NUMINAMATH_CALUDE_parabola_y_range_l64_6491

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus-to-point distance
def focus_distance (y : ℝ) : ℝ := y + 2

-- Define the condition for intersection with directrix
def intersects_directrix (y : ℝ) : Prop := focus_distance y > 4

theorem parabola_y_range (x y : ℝ) :
  parabola x y → intersects_directrix y → y > 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_range_l64_6491


namespace NUMINAMATH_CALUDE_women_per_table_l64_6477

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 6 →
  men_per_table = 5 →
  total_customers = 48 →
  ∃ (women_per_table : ℕ),
    women_per_table * num_tables + men_per_table * num_tables = total_customers ∧
    women_per_table = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_women_per_table_l64_6477


namespace NUMINAMATH_CALUDE_even_decreasing_comparison_l64_6455

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

end NUMINAMATH_CALUDE_even_decreasing_comparison_l64_6455


namespace NUMINAMATH_CALUDE_matrix_power_101_l64_6440

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

end NUMINAMATH_CALUDE_matrix_power_101_l64_6440


namespace NUMINAMATH_CALUDE_inequality_proof_l64_6400

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l64_6400


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l64_6468

theorem cubic_sum_problem (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 2) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a * b * c = 1/6 ∧ a^4 + b^4 + c^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l64_6468


namespace NUMINAMATH_CALUDE_same_color_probability_l64_6489

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- Calculates the number of combinations of n items taken r at a time -/
def combinations (n r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

/-- The probability of drawing two balls of the same color -/
theorem same_color_probability : 
  (combinations white_balls drawn_balls + combinations red_balls drawn_balls) / 
  combinations total_balls drawn_balls = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l64_6489


namespace NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l64_6488

theorem parallel_resistors_combined_resistance :
  let r1 : ℚ := 2
  let r2 : ℚ := 5
  let r3 : ℚ := 6
  let r : ℚ := (1 / r1 + 1 / r2 + 1 / r3)⁻¹
  r = 15 / 13 := by sorry

end NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l64_6488


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l64_6476

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l64_6476


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1821_l64_6463

theorem smallest_prime_factor_of_1821 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1821 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1821 → p ≤ q :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1821_l64_6463


namespace NUMINAMATH_CALUDE_coin_game_probability_l64_6496

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing three coins -/
structure ThreeCoinToss :=
  (first second third : CoinOutcome)

/-- Defines a winning outcome in the Coin Game -/
def is_winning_toss (toss : ThreeCoinToss) : Prop :=
  (toss.first = CoinOutcome.Heads ∧ toss.second = CoinOutcome.Heads ∧ toss.third = CoinOutcome.Heads) ∨
  (toss.first = CoinOutcome.Tails ∧ toss.second = CoinOutcome.Tails ∧ toss.third = CoinOutcome.Tails)

/-- The set of all possible outcomes when tossing three coins -/
def all_outcomes : Finset ThreeCoinToss := sorry

/-- The set of winning outcomes in the Coin Game -/
def winning_outcomes : Finset ThreeCoinToss := sorry

/-- Theorem stating that the probability of winning the Coin Game is 1/4 -/
theorem coin_game_probability : 
  (Finset.card winning_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_game_probability_l64_6496


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l64_6416

theorem unique_quadratic_solution (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → m = 0 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l64_6416


namespace NUMINAMATH_CALUDE_amelia_half_money_left_l64_6430

/-- Represents the fraction of money Amelia has left after buying all books -/
def amelia_money_left (total_money : ℝ) (book_cost : ℝ) (num_books : ℕ) : ℝ :=
  total_money - (book_cost * num_books)

/-- Theorem stating that Amelia will have half of her money left after buying all books -/
theorem amelia_half_money_left 
  (total_money : ℝ) (book_cost : ℝ) (num_books : ℕ) 
  (h1 : total_money > 0) 
  (h2 : book_cost > 0) 
  (h3 : num_books > 0)
  (h4 : (1/4) * total_money = (1/2) * (book_cost * num_books)) :
  amelia_money_left total_money book_cost num_books = (1/2) * total_money := by
  sorry

#check amelia_half_money_left

end NUMINAMATH_CALUDE_amelia_half_money_left_l64_6430


namespace NUMINAMATH_CALUDE_container_capacity_l64_6428

/-- Given that 8 liters is 20% of a container's capacity, prove that 40 such containers have a total capacity of 1600 liters. -/
theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (h2 : container_capacity > 0) : 
  40 * container_capacity = 1600 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l64_6428


namespace NUMINAMATH_CALUDE_max_dot_product_l64_6415

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, -1)

-- Define the moving point M
def M : Set (ℝ × ℝ) := {p | -2 ≤ p.1 ∧ p.1 ≤ 2 ∧ -2 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem max_dot_product :
  ∃ (max : ℝ), max = 4 ∧ ∀ m ∈ M, dot_product (m.1 - O.1, m.2 - O.2) (C.1 - O.1, C.2 - O.2) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l64_6415


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_153_l64_6471

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

end NUMINAMATH_CALUDE_quadrilateral_area_is_153_l64_6471


namespace NUMINAMATH_CALUDE_composite_equal_if_same_greatest_divisors_l64_6469

/-- The set of greatest divisors of a natural number, excluding the number itself -/
def greatestDivisors (n : ℕ) : Set ℕ :=
  {d | d ∣ n ∧ d ≠ n ∧ ∀ k, k ∣ n ∧ k ≠ n → k ≤ d}

/-- Two natural numbers are composite if they are greater than 1 and not prime -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

theorem composite_equal_if_same_greatest_divisors (a b : ℕ) 
    (ha : isComposite a) (hb : isComposite b) 
    (h : greatestDivisors a = greatestDivisors b) : 
  a = b := by
  sorry

end NUMINAMATH_CALUDE_composite_equal_if_same_greatest_divisors_l64_6469


namespace NUMINAMATH_CALUDE_stocker_wait_time_l64_6494

def total_shopping_time : ℕ := 90
def shopping_time : ℕ := 42
def cart_wait_time : ℕ := 3
def employee_wait_time : ℕ := 13
def checkout_wait_time : ℕ := 18

theorem stocker_wait_time :
  total_shopping_time - shopping_time - (cart_wait_time + employee_wait_time + checkout_wait_time) = 14 := by
  sorry

end NUMINAMATH_CALUDE_stocker_wait_time_l64_6494


namespace NUMINAMATH_CALUDE_greatest_possible_N_l64_6484

theorem greatest_possible_N : ∃ (N : ℕ), 
  (N = 5) ∧ 
  (∀ k : ℕ, k > 5 → ¬∃ (S : Finset ℕ), 
    (Finset.card S = 2^k - 1) ∧ 
    (∀ x y : ℕ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 2014)) ∧
  (∃ (S : Finset ℕ), 
    (Finset.card S = 2^5 - 1) ∧ 
    (∀ x y : ℕ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 2014)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_N_l64_6484


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l64_6445

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

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l64_6445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l64_6474

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  a 3 + a 4 + a 5 + a 6 + a 7 = 45 →
  a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l64_6474


namespace NUMINAMATH_CALUDE_quadratic_always_real_root_l64_6482

theorem quadratic_always_real_root (b : ℝ) : 
  ∃ x : ℝ, x^2 + b*x - 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_real_root_l64_6482


namespace NUMINAMATH_CALUDE_eight_divided_by_recurring_third_l64_6439

theorem eight_divided_by_recurring_third (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_recurring_third_l64_6439


namespace NUMINAMATH_CALUDE_mountaineering_teams_l64_6472

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

end NUMINAMATH_CALUDE_mountaineering_teams_l64_6472


namespace NUMINAMATH_CALUDE_smallest_k_with_odd_solutions_l64_6490

/-- The number of positive integral solutions to the equation 2xy - 3x - 5y = k -/
def num_solutions (k : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 * p.2 - 3 * p.1 - 5 * p.2 = k) (Finset.product (Finset.range 1000) (Finset.range 1000))).card

/-- Predicate to check if a number is odd -/
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_k_with_odd_solutions :
  (∀ k < 5, ¬(is_odd (num_solutions k))) ∧ 
  (is_odd (num_solutions 5)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_with_odd_solutions_l64_6490


namespace NUMINAMATH_CALUDE_no_k_for_all_positive_quadratic_l64_6456

theorem no_k_for_all_positive_quadratic : ¬∃ k : ℝ, ∀ x : ℝ, x^2 - (k - 4)*x - (k + 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_k_for_all_positive_quadratic_l64_6456


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l64_6492

theorem max_sum_of_factors (x y : ℕ+) (h : x * y = 48) : 
  ∃ (a b : ℕ+), a * b = 48 ∧ a + b ≤ x + y ∧ a + b = 49 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l64_6492


namespace NUMINAMATH_CALUDE_group_contains_perfect_square_diff_l64_6411

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem group_contains_perfect_square_diff :
  ∀ (partition : Fin 3 → Set ℕ),
    (∀ n : ℕ, n ≤ 46 → ∃ i : Fin 3, n ∈ partition i) →
    (∀ i j : Fin 3, i ≠ j → partition i ∩ partition j = ∅) →
    (∀ i : Fin 3, partition i ⊆ Finset.range 47) →
    ∃ (i : Fin 3) (a b : ℕ), 
      a ∈ partition i ∧ 
      b ∈ partition i ∧ 
      a ≠ b ∧ 
      is_perfect_square (max a b - min a b) :=
by
  sorry

#check group_contains_perfect_square_diff

end NUMINAMATH_CALUDE_group_contains_perfect_square_diff_l64_6411


namespace NUMINAMATH_CALUDE_house_number_unit_digit_l64_6432

def is_divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def hundred_digit (n : ℕ) : ℕ := (n / 100) % 10

def unit_digit (n : ℕ) : ℕ := n % 10

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem house_number_unit_digit (n : ℕ) 
  (three_digit : 100 ≤ n ∧ n < 1000)
  (exactly_three_true : ∃ (s1 s2 s3 s4 s5 : Prop), 
    (s1 = is_divisible_by n 9) ∧
    (s2 = is_even n) ∧
    (s3 = (hundred_digit n = 3)) ∧
    (s4 = is_odd (unit_digit n)) ∧
    (s5 = is_divisible_by n 5) ∧
    ((s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ ¬s5) ∨
     (s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ s5) ∨
     (s1 ∧ s2 ∧ ¬s3 ∧ ¬s4 ∧ s5) ∨
     (s1 ∧ ¬s2 ∧ s3 ∧ ¬s4 ∧ s5) ∨
     (¬s1 ∧ s2 ∧ s3 ∧ ¬s4 ∧ s5))) :
  unit_digit n = 0 := by sorry

end NUMINAMATH_CALUDE_house_number_unit_digit_l64_6432


namespace NUMINAMATH_CALUDE_projection_problem_l64_6495

def vector1 : ℝ × ℝ := (3, -2)
def vector2 : ℝ × ℝ := (2, 5)

def is_projection (v p : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), p = (k * v.1, k * v.2)

theorem projection_problem (v : ℝ × ℝ) (p : ℝ × ℝ) :
  is_projection v p ∧ is_projection v p →
  p = (133/50, 49/50) := by sorry

end NUMINAMATH_CALUDE_projection_problem_l64_6495


namespace NUMINAMATH_CALUDE_function_value_at_symmetry_point_l64_6424

/-- Given a function f(x) = 3cos(ωx + φ) that satisfies f(π/6 + x) = f(π/6 - x) for all x,
    prove that f(π/6) equals either 3 or -3 -/
theorem function_value_at_symmetry_point 
  (ω φ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = 3 * Real.cos (ω * x + φ))
  (h2 : ∀ x, f (π/6 + x) = f (π/6 - x)) :
  f (π/6) = 3 ∨ f (π/6) = -3 :=
sorry

end NUMINAMATH_CALUDE_function_value_at_symmetry_point_l64_6424


namespace NUMINAMATH_CALUDE_sum_of_multiples_l64_6458

theorem sum_of_multiples (m n : ℝ) : 2 * m + 3 * n = 2*m + 3*n := by sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l64_6458


namespace NUMINAMATH_CALUDE_equal_value_proof_l64_6446

theorem equal_value_proof (a b : ℝ) (h1 : 10 * a = 6 * b) (h2 : 120 * a * b = 800) :
  10 * a = 20 ∧ 6 * b = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_value_proof_l64_6446


namespace NUMINAMATH_CALUDE_robins_hair_length_l64_6473

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

end NUMINAMATH_CALUDE_robins_hair_length_l64_6473


namespace NUMINAMATH_CALUDE_star_running_back_yards_l64_6429

/-- Represents the yardage statistics for a football player -/
structure PlayerStats where
  total_yards : ℕ
  pass_yards : ℕ
  run_yards : ℕ

/-- Calculates the running yards for a player given total yards and pass yards -/
def calculate_run_yards (total : ℕ) (pass : ℕ) : ℕ :=
  total - pass

/-- Theorem stating that the star running back's running yards is 90 -/
theorem star_running_back_yards (player : PlayerStats)
    (h1 : player.total_yards = 150)
    (h2 : player.pass_yards = 60)
    (h3 : player.run_yards = calculate_run_yards player.total_yards player.pass_yards) :
    player.run_yards = 90 := by
  sorry

end NUMINAMATH_CALUDE_star_running_back_yards_l64_6429


namespace NUMINAMATH_CALUDE_always_three_same_color_sum_zero_l64_6441

-- Define a type for colors
inductive Color
| White
| Black

-- Define a function type for coloring integers
def Coloring := Int → Color

-- Define the property that 2016 and 2017 are different colors
def DifferentColors (c : Coloring) : Prop :=
  c 2016 ≠ c 2017

-- Define the property of three integers having the same color and summing to zero
def ThreeSameColorSumZero (c : Coloring) : Prop :=
  ∃ x y z : Int, (c x = c y ∧ c y = c z) ∧ x + y + z = 0

-- State the theorem
theorem always_three_same_color_sum_zero (c : Coloring) :
  DifferentColors c → ThreeSameColorSumZero c := by
  sorry

end NUMINAMATH_CALUDE_always_three_same_color_sum_zero_l64_6441


namespace NUMINAMATH_CALUDE_yogurt_combinations_l64_6478

theorem yogurt_combinations (n_flavors : ℕ) (n_toppings : ℕ) : 
  n_flavors = 4 → n_toppings = 8 → 
  n_flavors * (n_toppings.choose 3) = 224 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l64_6478


namespace NUMINAMATH_CALUDE_dining_bill_proof_l64_6479

theorem dining_bill_proof (num_people : ℕ) (individual_payment : ℚ) (tip_percentage : ℚ) 
  (h1 : num_people = 7)
  (h2 : individual_payment = 21.842857142857145)
  (h3 : tip_percentage = 1/10) :
  (num_people : ℚ) * individual_payment / (1 + tip_percentage) = 139 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_proof_l64_6479


namespace NUMINAMATH_CALUDE_max_coprime_partition_l64_6402

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_partition (A B : Finset ℕ) : Prop :=
  (∀ a ∈ A, 2 ≤ a ∧ a ≤ 20) ∧
  (∀ b ∈ B, 2 ≤ b ∧ b ≤ 20) ∧
  (∀ a ∈ A, ∀ b ∈ B, is_coprime a b) ∧
  A ∩ B = ∅ ∧
  A ∪ B ⊆ Finset.range 19 ∪ {20}

theorem max_coprime_partition :
  ∃ A B : Finset ℕ,
    valid_partition A B ∧
    A.card * B.card = 49 ∧
    ∀ C D : Finset ℕ, valid_partition C D → C.card * D.card ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_coprime_partition_l64_6402


namespace NUMINAMATH_CALUDE_intersection_sum_l64_6460

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The cubic equation y = x³ - 3x - 4 -/
def cubic (p : Point) : Prop :=
  p.y = p.x^3 - 3*p.x - 4

/-- The linear equation x + 3y = 3 -/
def linear (p : Point) : Prop :=
  p.x + 3*p.y = 3

theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : Point),
    (cubic p₁ ∧ linear p₁) ∧
    (cubic p₂ ∧ linear p₂) ∧
    (cubic p₃ ∧ linear p₃) ∧
    (p₁.x + p₂.x + p₃.x = 8/3) ∧
    (p₁.y + p₂.y + p₃.y = 19/9) := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l64_6460


namespace NUMINAMATH_CALUDE_quiz_homework_difference_l64_6466

/-- Represents the points distribution in Paul's biology class -/
structure PointsDistribution where
  total : ℕ
  homework : ℕ
  quiz : ℕ
  test : ℕ

/-- The conditions for Paul's point distribution -/
def paulsDistribution (p : PointsDistribution) : Prop :=
  p.total = 265 ∧
  p.homework = 40 ∧
  p.test = 4 * p.quiz ∧
  p.total = p.homework + p.quiz + p.test

/-- Theorem stating the difference between quiz and homework points -/
theorem quiz_homework_difference (p : PointsDistribution) 
  (h : paulsDistribution p) : p.quiz - p.homework = 5 := by
  sorry

end NUMINAMATH_CALUDE_quiz_homework_difference_l64_6466
