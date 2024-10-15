import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l2924_292409

theorem fraction_simplification (a b c : ℝ) 
  (h1 : b + c + a ≠ 0) (h2 : b + c ≠ a) : 
  (b^2 + a^2 - c^2 + 2*b*c) / (b^2 + c^2 - a^2 + 2*b*c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2924_292409


namespace NUMINAMATH_CALUDE_negative_root_range_l2924_292495

theorem negative_root_range (x a : ℝ) : 
  x < 0 → 
  (2/3)^x = (1+a)/(1-a) → 
  0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_negative_root_range_l2924_292495


namespace NUMINAMATH_CALUDE_min_sum_squares_l2924_292466

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 3 * x₂ + 4 * x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 7200 / 13 ∧ 
  ∃ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧ 
    y₁ + 3 * y₂ + 4 * y₃ = 120 ∧ 
    y₁^2 + y₂^2 + y₃^2 = 7200 / 13 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2924_292466


namespace NUMINAMATH_CALUDE_chosen_number_proof_l2924_292483

theorem chosen_number_proof (x : ℝ) : (x / 2) - 100 = 4 → x = 208 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l2924_292483


namespace NUMINAMATH_CALUDE_pebbles_on_day_15_l2924_292417

/-- Represents Murtha's pebble collection strategy -/
def pebbleCollection (n : ℕ) : ℕ := 
  if n < 15 then n else 2 * 15

/-- The sum of pebbles collected up to day n -/
def totalPebbles (n : ℕ) : ℕ := 
  (List.range n).map pebbleCollection |>.sum

/-- Theorem stating the total number of pebbles collected by the end of the 15th day -/
theorem pebbles_on_day_15 : totalPebbles 15 = 135 := by
  sorry

end NUMINAMATH_CALUDE_pebbles_on_day_15_l2924_292417


namespace NUMINAMATH_CALUDE_arcsin_arccos_half_pi_l2924_292434

theorem arcsin_arccos_half_pi : 
  Real.arcsin (1/2) = π/6 ∧ Real.arccos (1/2) = π/3 := by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_half_pi_l2924_292434


namespace NUMINAMATH_CALUDE_marie_profit_l2924_292482

def total_loaves : ℕ := 60
def cost_per_loaf : ℚ := 1
def morning_price : ℚ := 3
def afternoon_discount : ℚ := 0.25
def donated_loaves : ℕ := 5

def morning_sales : ℕ := total_loaves / 3
def remaining_after_morning : ℕ := total_loaves - morning_sales
def afternoon_sales : ℕ := remaining_after_morning / 2
def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_sales
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

def afternoon_price : ℚ := morning_price * (1 - afternoon_discount)

def total_revenue : ℚ := morning_sales * morning_price + afternoon_sales * afternoon_price
def total_cost : ℚ := total_loaves * cost_per_loaf
def profit : ℚ := total_revenue - total_cost

theorem marie_profit : profit = 45 := by
  sorry

end NUMINAMATH_CALUDE_marie_profit_l2924_292482


namespace NUMINAMATH_CALUDE_fishing_problem_l2924_292498

theorem fishing_problem (total fish_jason fish_ryan fish_jeffery : ℕ) : 
  total = 100 ∧ 
  fish_ryan = 3 * fish_jason ∧ 
  fish_jeffery = 2 * fish_ryan ∧ 
  total = fish_jason + fish_ryan + fish_jeffery →
  fish_jeffery = 60 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l2924_292498


namespace NUMINAMATH_CALUDE_transfer_increases_averages_l2924_292428

/-- Represents a group of students with their total score and count -/
structure StudentGroup where
  totalScore : ℚ
  count : ℕ

/-- Calculates the average score of a group -/
def averageScore (group : StudentGroup) : ℚ :=
  group.totalScore / group.count

/-- Theorem: Transferring Lopatin and Filin increases average scores in both groups -/
theorem transfer_increases_averages
  (groupA groupB : StudentGroup)
  (lopatinScore filinScore : ℚ)
  (h1 : groupA.count = 10)
  (h2 : groupB.count = 10)
  (h3 : averageScore groupA = 47.2)
  (h4 : averageScore groupB = 41.8)
  (h5 : 41.8 < lopatinScore) (h6 : lopatinScore < 47.2)
  (h7 : 41.8 < filinScore) (h8 : filinScore < 47.2)
  (h9 : lopatinScore = 47)
  (h10 : filinScore = 44) :
  let newGroupA : StudentGroup := ⟨groupA.totalScore - lopatinScore - filinScore, 8⟩
  let newGroupB : StudentGroup := ⟨groupB.totalScore + lopatinScore + filinScore, 12⟩
  averageScore newGroupA > 47.5 ∧ averageScore newGroupB > 42.2 := by
  sorry

end NUMINAMATH_CALUDE_transfer_increases_averages_l2924_292428


namespace NUMINAMATH_CALUDE_cargo_transport_possible_l2924_292412

/-- Represents the cargo transportation problem -/
structure CargoTransport where
  totalCargo : ℕ
  bagCapacity : ℕ
  truckCapacity : ℕ
  maxTrips : ℕ

/-- Checks if the cargo can be transported within the given number of trips -/
def canTransport (ct : CargoTransport) : Prop :=
  ∃ (trips : ℕ), trips ≤ ct.maxTrips ∧ trips * ct.truckCapacity ≥ ct.totalCargo

/-- Theorem stating that 36 tons of cargo can be transported in 11 trips or fewer -/
theorem cargo_transport_possible : 
  canTransport ⟨36, 1, 4, 11⟩ := by
  sorry

#check cargo_transport_possible

end NUMINAMATH_CALUDE_cargo_transport_possible_l2924_292412


namespace NUMINAMATH_CALUDE_marias_car_trip_l2924_292429

theorem marias_car_trip (D : ℝ) : 
  D / 2 + (D - D / 2) / 4 + 210 = D → D = 560 := by sorry

end NUMINAMATH_CALUDE_marias_car_trip_l2924_292429


namespace NUMINAMATH_CALUDE_correct_fraction_l2924_292420

theorem correct_fraction (number : ℚ) (incorrect_fraction : ℚ) (difference : ℚ) :
  number = 96 →
  incorrect_fraction = 5 / 6 →
  incorrect_fraction * number = number * x + difference →
  difference = 50 →
  x = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_fraction_l2924_292420


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l2924_292481

theorem solutions_of_quadratic_equation :
  ∀ x : ℝ, (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_equation_l2924_292481


namespace NUMINAMATH_CALUDE_annas_gold_amount_annas_gold_theorem_l2924_292403

theorem annas_gold_amount (gary_gold : ℕ) (gary_cost_per_gram : ℕ) (anna_cost_per_gram : ℕ) (total_cost : ℕ) : ℕ :=
  let gary_total_cost := gary_gold * gary_cost_per_gram
  let anna_total_cost := total_cost - gary_total_cost
  anna_total_cost / anna_cost_per_gram

theorem annas_gold_theorem :
  annas_gold_amount 30 15 20 1450 = 50 := by
  sorry

end NUMINAMATH_CALUDE_annas_gold_amount_annas_gold_theorem_l2924_292403


namespace NUMINAMATH_CALUDE_probability_three_and_zero_painted_faces_l2924_292411

/-- Represents a 5x5x5 cube with three faces sharing a common corner painted red -/
structure PaintedCube :=
  (side_length : ℕ)
  (total_cubes : ℕ)
  (painted_faces : ℕ)

/-- Counts the number of unit cubes with a specific number of painted faces -/
def count_painted_cubes (c : PaintedCube) (num_painted_faces : ℕ) : ℕ :=
  sorry

/-- Calculates the probability of selecting two specific types of unit cubes -/
def probability_of_selection (c : PaintedCube) (faces1 faces2 : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of selecting one cube with three
    painted faces and one cube with no painted faces -/
theorem probability_three_and_zero_painted_faces (c : PaintedCube) :
  c.side_length = 5 ∧ c.total_cubes = 125 ∧ c.painted_faces = 3 →
  probability_of_selection c 3 0 = 44 / 3875 :=
sorry

end NUMINAMATH_CALUDE_probability_three_and_zero_painted_faces_l2924_292411


namespace NUMINAMATH_CALUDE_tan_alpha_equals_three_l2924_292418

theorem tan_alpha_equals_three (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sin α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10) : 
  Real.tan α = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_three_l2924_292418


namespace NUMINAMATH_CALUDE_vector_collinearity_l2924_292484

theorem vector_collinearity (m n : ℝ) (h : n ≠ 0) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, 3]
  (∃ (k : ℝ), k ≠ 0 ∧ (fun i => m * a i - n * b i) = (fun i => k * (a i + 2 * b i))) →
  m / n = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2924_292484


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l2924_292450

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(24 ∣ m^2 ∧ 864 ∣ m^3)) ∧ 
  (24 ∣ n^2) ∧ (864 ∣ n^3) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l2924_292450


namespace NUMINAMATH_CALUDE_parabola_hyperbola_coincidence_l2924_292439

/-- The value of p for which the focus of the parabola y^2 = 2px coincides with
    the right vertex of the hyperbola x^2/4 - y^2 = 1 -/
theorem parabola_hyperbola_coincidence (p : ℝ) : 
  (∃ x y : ℝ, y^2 = 2*p*x ∧ x^2/4 - y^2 = 1 ∧ x = p/2 ∧ y = 0) → p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_coincidence_l2924_292439


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2924_292489

/-- The area of the shaded region in a geometric figure with the following properties:
    - A large square with side length 20 cm
    - Four quarter circles with radius 10 cm centered at the corners of the large square
    - A smaller square with side length 10 cm centered inside the larger square
    is equal to 100π - 100 cm². -/
theorem shaded_area_calculation (π : ℝ) : ℝ := by
  -- Define the side lengths and radius
  let large_square_side : ℝ := 20
  let small_square_side : ℝ := 10
  let quarter_circle_radius : ℝ := 10

  -- Define the areas
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_area : ℝ := small_square_side ^ 2
  let quarter_circles_area : ℝ := π * quarter_circle_radius ^ 2

  -- Calculate the shaded area
  let shaded_area : ℝ := quarter_circles_area - small_square_area

  -- Prove that the shaded area equals 100π - 100
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2924_292489


namespace NUMINAMATH_CALUDE_performance_arrangements_l2924_292445

-- Define the number of singers
def n : ℕ := 6

-- Define the number of singers with specific order requirements (A, B, C)
def k : ℕ := 3

-- Define the number of valid orders for B and C relative to A
def valid_orders : ℕ := 4

-- Theorem statement
theorem performance_arrangements : 
  (valid_orders : ℕ) * (n.factorial / k.factorial) = 480 := by
  sorry

end NUMINAMATH_CALUDE_performance_arrangements_l2924_292445


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l2924_292491

theorem fahrenheit_to_celsius (C F : ℚ) : 
  C = 35 → C = (7/12) * (F - 40) → F = 100 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l2924_292491


namespace NUMINAMATH_CALUDE_simplify_expression_l2924_292452

theorem simplify_expression (x y : ℝ) : (3*x)^4 + (4*x)*(x^3) + (5*y)^2 = 85*x^4 + 25*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2924_292452


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l2924_292440

theorem x_squared_plus_y_squared_equals_four
  (x y : ℝ)
  (h1 : x^3 = 3*y^2*x + 5 - Real.sqrt 7)
  (h2 : y^3 = 3*x^2*y + 5 + Real.sqrt 7) :
  x^2 + y^2 = 4 := by sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l2924_292440


namespace NUMINAMATH_CALUDE_seedling_survival_probability_l2924_292497

/-- Represents the data for a single sample of transplanted ginkgo seedlings -/
structure SeedlingData where
  transplanted : ℕ
  survived : ℕ
  survival_rate : ℚ
  transplanted_positive : transplanted > 0
  survived_le_transplanted : survived ≤ transplanted
  rate_calculation : survival_rate = survived / transplanted

/-- The data set of ginkgo seedling transplantation experiments -/
def seedling_samples : List SeedlingData := [
  ⟨100, 84, 84/100, by norm_num, by norm_num, by norm_num⟩,
  ⟨300, 279, 279/300, by norm_num, by norm_num, by norm_num⟩,
  ⟨600, 505, 505/600, by norm_num, by norm_num, by norm_num⟩,
  ⟨1000, 847, 847/1000, by norm_num, by norm_num, by norm_num⟩,
  ⟨7000, 6337, 6337/7000, by norm_num, by norm_num, by norm_num⟩,
  ⟨15000, 13581, 13581/15000, by norm_num, by norm_num, by norm_num⟩
]

/-- The estimated probability of ginkgo seedling survival -/
def estimated_probability : ℚ := 9/10

/-- Theorem stating that the estimated probability approaches 0.9 as sample size increases -/
theorem seedling_survival_probability :
  ∀ ε > 0, ∃ N, ∀ sample ∈ seedling_samples,
    sample.transplanted ≥ N →
    |sample.survival_rate - estimated_probability| < ε :=
sorry

end NUMINAMATH_CALUDE_seedling_survival_probability_l2924_292497


namespace NUMINAMATH_CALUDE_lance_reading_plan_l2924_292485

/-- Represents the number of pages read on each day -/
structure ReadingPlan where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Checks if a reading plan is valid according to the given conditions -/
def isValidPlan (plan : ReadingPlan) (totalPages : ℕ) : Prop :=
  plan.day2 = plan.day1 - 5 ∧
  plan.day3 = 35 ∧
  plan.day1 + plan.day2 + plan.day3 = totalPages

theorem lance_reading_plan (totalPages : ℕ) (h : totalPages = 100) :
  ∃ (plan : ReadingPlan), isValidPlan plan totalPages ∧ plan.day1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_lance_reading_plan_l2924_292485


namespace NUMINAMATH_CALUDE_C_excircle_touches_circumcircle_l2924_292402

-- Define the basic geometric structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Circle where
  center : Point
  radius : ℝ

-- Define the semiperimeter of a triangle
def semiperimeter (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the C-excircle of a triangle
def C_excircle (t : Triangle) : Circle := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Circle := sorry

-- Define tangency between two circles
def are_tangent (c1 c2 : Circle) : Prop := sorry

-- Theorem statement
theorem C_excircle_touches_circumcircle 
  (ABC : Triangle) 
  (p : ℝ) 
  (E F : Point) :
  semiperimeter ABC = p →
  E.x ≤ F.x →
  distance ABC.A E + distance E F + distance F ABC.B = distance ABC.A ABC.B →
  distance ABC.C E = p →
  distance ABC.C F = p →
  are_tangent (C_excircle ABC) (circumcircle (Triangle.mk E F ABC.C)) :=
sorry

end NUMINAMATH_CALUDE_C_excircle_touches_circumcircle_l2924_292402


namespace NUMINAMATH_CALUDE_factorial_sum_equation_l2924_292474

theorem factorial_sum_equation : ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧ S.sum id = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equation_l2924_292474


namespace NUMINAMATH_CALUDE_crabapple_theorem_l2924_292422

/-- The number of different sequences of crabapple recipients in a week for two classes -/
def crabapple_sequences (students1 : ℕ) (meetings1 : ℕ) (students2 : ℕ) (meetings2 : ℕ) : ℕ :=
  (students1 ^ meetings1) * (students2 ^ meetings2)

/-- Theorem stating the number of crabapple recipient sequences for the given classes -/
theorem crabapple_theorem :
  crabapple_sequences 12 3 9 2 = 139968 := by
  sorry

#eval crabapple_sequences 12 3 9 2

end NUMINAMATH_CALUDE_crabapple_theorem_l2924_292422


namespace NUMINAMATH_CALUDE_contrapositive_product_nonzero_l2924_292487

theorem contrapositive_product_nonzero (a b : ℝ) :
  (¬(a * b ≠ 0) → ¬(a ≠ 0 ∧ b ≠ 0)) ↔ ((a = 0 ∨ b = 0) → a * b = 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_product_nonzero_l2924_292487


namespace NUMINAMATH_CALUDE_gcd_1681_1705_l2924_292407

theorem gcd_1681_1705 : Nat.gcd 1681 1705 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1681_1705_l2924_292407


namespace NUMINAMATH_CALUDE_room_length_proof_l2924_292453

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 6 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 25650 →
  paving_rate = 900 →
  (total_cost / paving_rate) / width = 6 := by
sorry

end NUMINAMATH_CALUDE_room_length_proof_l2924_292453


namespace NUMINAMATH_CALUDE_solve_system_l2924_292441

theorem solve_system (x y : ℚ) (h1 : x - y = 12) (h2 : 2 * x + y = 10) : y = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2924_292441


namespace NUMINAMATH_CALUDE_ones_digit_of_33_power_l2924_292424

theorem ones_digit_of_33_power (n : ℕ) : 
  (33^(33*(12^12))) % 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_ones_digit_of_33_power_l2924_292424


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l2924_292431

theorem sqrt_six_times_sqrt_two_equals_two_sqrt_three :
  Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l2924_292431


namespace NUMINAMATH_CALUDE_set_equality_problem_l2924_292486

theorem set_equality_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 4} →
  B = {0, 1, a} →
  A ∪ B = {0, 1, 4} →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_set_equality_problem_l2924_292486


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l2924_292444

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 59405 / 30958 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l2924_292444


namespace NUMINAMATH_CALUDE_print_shop_charge_l2924_292442

/-- The charge per color copy at print shop X -/
def charge_x : ℝ := 1.25

/-- The charge per color copy at print shop Y -/
def charge_y : ℝ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 80

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℝ := 120

theorem print_shop_charge : 
  charge_x * num_copies + additional_charge = charge_y * num_copies := by
  sorry

#check print_shop_charge

end NUMINAMATH_CALUDE_print_shop_charge_l2924_292442


namespace NUMINAMATH_CALUDE_initial_salt_percentage_l2924_292404

theorem initial_salt_percentage
  (initial_mass : ℝ)
  (added_salt : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_mass = 100)
  (h2 : added_salt = 38.46153846153846)
  (h3 : final_percentage = 35) :
  let final_mass := initial_mass + added_salt
  let final_salt_mass := (final_percentage / 100) * final_mass
  let initial_salt_mass := final_salt_mass - added_salt
  initial_salt_mass / initial_mass * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_initial_salt_percentage_l2924_292404


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2924_292425

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Main theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_product : a 2 * a 4 = 4) :
  a 1 * a 5 + a 3 = 6 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2924_292425


namespace NUMINAMATH_CALUDE_marissa_boxes_tied_l2924_292410

/-- The number of boxes tied with a given amount of ribbon -/
def boxes_tied (total_ribbon leftover_ribbon ribbon_per_box : ℚ) : ℚ :=
  (total_ribbon - leftover_ribbon) / ribbon_per_box

/-- Theorem: Given the conditions, Marissa tied 5 boxes -/
theorem marissa_boxes_tied :
  boxes_tied 4.5 1 0.7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_marissa_boxes_tied_l2924_292410


namespace NUMINAMATH_CALUDE_units_digit_product_minus_cube_l2924_292449

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_product_minus_cube : units_digit (8 * 18 * 1998 - 8^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_minus_cube_l2924_292449


namespace NUMINAMATH_CALUDE_total_bread_is_370_l2924_292490

/-- The amount of bread Cara ate for dinner, in grams -/
def dinner_bread : ℕ := 240

/-- The amount of bread Cara ate for lunch, in grams -/
def lunch_bread : ℕ := dinner_bread / 8

/-- The amount of bread Cara ate for breakfast, in grams -/
def breakfast_bread : ℕ := dinner_bread / 6

/-- The amount of bread Cara ate for snack, in grams -/
def snack_bread : ℕ := dinner_bread / 4

/-- The total amount of bread Cara ate, in grams -/
def total_bread : ℕ := dinner_bread + lunch_bread + breakfast_bread + snack_bread

theorem total_bread_is_370 : total_bread = 370 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_is_370_l2924_292490


namespace NUMINAMATH_CALUDE_curve_and_lines_distance_properties_l2924_292479

/-- Given a curve C and lines l and l1 in a 2D plane, prove properties about distances -/
theorem curve_and_lines_distance_properties
  (B : ℝ × ℝ)
  (C : ℝ → ℝ × ℝ)
  (A : ℝ × ℝ)
  (l l1 : ℝ × ℝ → Prop)
  (h_B : B = (1, 1))
  (h_C : ∀ θ, C θ = (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ))
  (h_A : A = (4 * Real.sqrt 2 * Real.cos (π/4), 4 * Real.sqrt 2 * Real.sin (π/4)))
  (h_l : ∃ a, ∀ ρ θ, l (ρ * Real.cos θ, ρ * Real.sin θ) ↔ ρ * Real.cos (θ - π/4) = a)
  (h_l_A : l A)
  (h_l1_parallel : ∃ k, ∀ p, l1 p ↔ l (p.1 - k, p.2 - k))
  (h_l1_B : l1 B)
  (h_l1_intersect : ∃ M N, M ≠ N ∧ l1 (C M) ∧ l1 (C N)) :
  (∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 ∧
    ∀ p, (∃ θ, C θ = p) → 
      ∀ q, l q → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_curve_and_lines_distance_properties_l2924_292479


namespace NUMINAMATH_CALUDE_square_field_side_length_l2924_292472

theorem square_field_side_length (area : ℝ) (side : ℝ) :
  area = 256 → side ^ 2 = area → side = 16 := by sorry

end NUMINAMATH_CALUDE_square_field_side_length_l2924_292472


namespace NUMINAMATH_CALUDE_number_ordering_l2924_292447

theorem number_ordering : 7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2924_292447


namespace NUMINAMATH_CALUDE_same_gender_probability_theorem_l2924_292496

/-- Represents a school with a certain number of male and female teachers -/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- The probability of selecting two teachers of the same gender from two schools -/
def same_gender_probability (school_a school_b : School) : ℚ :=
  let total_combinations := (school_a.male_teachers + school_a.female_teachers) * (school_b.male_teachers + school_b.female_teachers)
  let same_gender_combinations := school_a.male_teachers * school_b.male_teachers + school_a.female_teachers * school_b.female_teachers
  same_gender_combinations / total_combinations

/-- Theorem stating that the probability of selecting two teachers of the same gender
    from the given schools is 4/9 -/
theorem same_gender_probability_theorem :
  let school_a := School.mk 2 1
  let school_b := School.mk 1 2
  same_gender_probability school_a school_b = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_gender_probability_theorem_l2924_292496


namespace NUMINAMATH_CALUDE_head_start_calculation_l2924_292478

/-- Proves that the head start given by A to B is 72 meters in a 96-meter race,
    given that A runs 4 times as fast as B and they finish at the same time. -/
theorem head_start_calculation (v_B : ℝ) (d : ℝ) 
  (h1 : v_B > 0)  -- B's speed is positive
  (h2 : 96 > d)   -- The head start is less than the total race distance
  (h3 : 96 / (4 * v_B) = (96 - d) / v_B)  -- A and B finish at the same time
  : d = 72 := by
  sorry

end NUMINAMATH_CALUDE_head_start_calculation_l2924_292478


namespace NUMINAMATH_CALUDE_one_third_in_one_sixth_l2924_292427

theorem one_third_in_one_sixth :
  (1 : ℚ) / 6 / ((1 : ℚ) / 3) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_one_third_in_one_sixth_l2924_292427


namespace NUMINAMATH_CALUDE_reflected_triangle_angles_l2924_292458

-- Define the triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the reflection operation
def reflect (t : Triangle) : Triangle := sorry

-- Define the property of being acute
def is_acute (t : Triangle) : Prop := sorry

-- Define the property of being scalene
def is_scalene (t : Triangle) : Prop := sorry

-- Define the property of A₁B₁C₁ being inside ABC
def is_inside (t1 t2 : Triangle) : Prop := sorry

-- Define the property of A₂B₂C₂ being outside ABC
def is_outside (t1 t2 : Triangle) : Prop := sorry

-- Define the theorem
theorem reflected_triangle_angles 
  (ABC : Triangle) 
  (A₁B₁C₁ : Triangle) 
  (A₂B₂C₂ : Triangle) :
  is_acute ABC →
  is_scalene ABC →
  is_inside A₁B₁C₁ ABC →
  is_outside A₂B₂C₂ ABC →
  (A₂B₂C₂.a = 20 ∨ A₂B₂C₂.b = 20 ∨ A₂B₂C₂.c = 20) →
  (A₂B₂C₂.a = 70 ∨ A₂B₂C₂.b = 70 ∨ A₂B₂C₂.c = 70) →
  ((A₁B₁C₁.a = 60 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 60) ∨
   (A₁B₁C₁.a = 140/3 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 220/3) ∨
   (A₁B₁C₁.a = 60 ∧ A₁B₁C₁.b = 140/3 ∧ A₁B₁C₁.c = 220/3) ∨
   (A₁B₁C₁.a = 220/3 ∧ A₁B₁C₁.b = 60 ∧ A₁B₁C₁.c = 140/3)) :=
by sorry

end NUMINAMATH_CALUDE_reflected_triangle_angles_l2924_292458


namespace NUMINAMATH_CALUDE_eldest_child_age_l2924_292493

/-- Represents the ages of three grandchildren -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.middle = ages.youngest + 3 ∧
  ages.eldest = 3 * ages.youngest ∧
  ages.eldest = ages.youngest + ages.middle + 2

/-- The theorem stating that the eldest child's age is 15 years -/
theorem eldest_child_age (ages : GrandchildrenAges) :
  satisfiesConditions ages → ages.eldest = 15 := by
  sorry


end NUMINAMATH_CALUDE_eldest_child_age_l2924_292493


namespace NUMINAMATH_CALUDE_cube_volume_division_l2924_292461

theorem cube_volume_division (V : ℝ) (a b : ℝ) (h1 : V > 0) (h2 : b > a) (h3 : a > 0) :
  let diagonal_ratio := a / (b - a)
  let volume_ratio := a^3 / (b^3 - a^3)
  ∃ (V1 V2 : ℝ), V1 + V2 = V ∧ V1 / V2 = volume_ratio :=
sorry

end NUMINAMATH_CALUDE_cube_volume_division_l2924_292461


namespace NUMINAMATH_CALUDE_clock_hand_alignments_in_day_l2924_292421

/-- Represents a traditional 12-hour analog clock -/
structure AnalogClock where
  hourHand : ℝ
  minuteHand : ℝ
  secondHand : ℝ

/-- The number of times the clock hands align in a 12-hour period -/
def alignmentsIn12Hours : ℕ := 1

/-- The number of 12-hour periods in a day -/
def periodsInDay : ℕ := 2

/-- Theorem: The number of times all three hands align in a 24-hour period is 2 -/
theorem clock_hand_alignments_in_day :
  alignmentsIn12Hours * periodsInDay = 2 := by sorry

end NUMINAMATH_CALUDE_clock_hand_alignments_in_day_l2924_292421


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2924_292499

-- Define the condition for a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m - 2) = 1

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ Set.Ioo (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2924_292499


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l2924_292465

/-- Represents a rectangular field with given area and width-length relationship -/
structure RectangularField where
  area : ℕ
  width_length_diff : ℕ

/-- Checks if the given length and width satisfy the conditions for a rectangular field -/
def is_valid_dimensions (field : RectangularField) (length width : ℕ) : Prop :=
  length * width = field.area ∧ length = width + field.width_length_diff

theorem rectangular_field_dimensions (field : RectangularField) 
  (h : field.area = 864 ∧ field.width_length_diff = 12) :
  ∃ (length width : ℕ), is_valid_dimensions field length width ∧ length = 36 ∧ width = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l2924_292465


namespace NUMINAMATH_CALUDE_solution_volume_proof_l2924_292446

-- Define the volume of pure acid in liters
def pure_acid_volume : ℝ := 1.6

-- Define the concentration of the solution as a percentage
def solution_concentration : ℝ := 20

-- Define the total volume of the solution in liters
def total_solution_volume : ℝ := 8

-- Theorem to prove
theorem solution_volume_proof :
  pure_acid_volume = (solution_concentration / 100) * total_solution_volume :=
by sorry

end NUMINAMATH_CALUDE_solution_volume_proof_l2924_292446


namespace NUMINAMATH_CALUDE_boat_speed_l2924_292468

theorem boat_speed (current_speed : ℝ) (downstream_distance : ℝ) (time : ℝ) :
  current_speed = 3 →
  downstream_distance = 6.75 →
  time = 0.25 →
  ∃ (boat_speed : ℝ),
    boat_speed = 24 ∧
    downstream_distance = (boat_speed + current_speed) * time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l2924_292468


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2924_292455

theorem contrapositive_equivalence (f : ℝ → ℝ) (a : ℝ) :
  (a ≥ (1/2) → ∀ x ≥ 0, f x ≥ 0) ↔
  (∃ x ≥ 0, f x < 0 → a < (1/2)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2924_292455


namespace NUMINAMATH_CALUDE_triangle_area_l2924_292494

theorem triangle_area (base height : ℝ) (h1 : base = 4.5) (h2 : height = 6) :
  (base * height) / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2924_292494


namespace NUMINAMATH_CALUDE_larger_number_proof_l2924_292456

theorem larger_number_proof (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2924_292456


namespace NUMINAMATH_CALUDE_highest_score_is_103_l2924_292436

def base_score : ℕ := 100

def score_adjustments : List ℤ := [3, -8, 0]

def actual_scores : List ℕ := score_adjustments.map (λ x => (base_score : ℤ) + x |>.toNat)

theorem highest_score_is_103 : actual_scores.maximum? = some 103 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_is_103_l2924_292436


namespace NUMINAMATH_CALUDE_unique_non_range_value_l2924_292415

/-- The function g defined as (px + q) / (rx + s) -/
noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

/-- The theorem stating that 30 is the unique number not in the range of g -/
theorem unique_non_range_value
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_11 : g p q r s 11 = 11)
  (h_41 : g p q r s 41 = 41)
  (h_inverse : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, ∀ x, g p q r s x ≠ y ∧ y = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_non_range_value_l2924_292415


namespace NUMINAMATH_CALUDE_complex_expression_equals_nine_l2924_292413

theorem complex_expression_equals_nine :
  (0.4 + 8 * (5 - 0.8 * (5 / 8)) - 5 / (2 + 1 / 2)) /
  ((1 + 7 / 8) * 8 - (8.9 - 2.6 / (2 / 3))) * (34 + 2 / 5) * 90 = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_nine_l2924_292413


namespace NUMINAMATH_CALUDE_ratio_is_five_thirds_l2924_292464

/-- Given a diagram with triangles, some shaded and some unshaded -/
structure TriangleDiagram where
  shaded : ℕ
  unshaded : ℕ

/-- The ratio of shaded to unshaded triangles -/
def shaded_unshaded_ratio (d : TriangleDiagram) : ℚ :=
  d.shaded / d.unshaded

theorem ratio_is_five_thirds (d : TriangleDiagram) 
  (h1 : d.shaded = 5) 
  (h2 : d.unshaded = 3) : 
  shaded_unshaded_ratio d = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_five_thirds_l2924_292464


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2924_292451

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2924_292451


namespace NUMINAMATH_CALUDE_fiftiethTerm_l2924_292476

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftiethTerm : arithmeticSequenceTerm 2 5 50 = 247 := by
  sorry

end NUMINAMATH_CALUDE_fiftiethTerm_l2924_292476


namespace NUMINAMATH_CALUDE_parabola_translation_l2924_292432

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 2 0 0  -- y = 2x²
  let translated := translate original 3 4
  y = translated.a * x^2 + translated.b * x + translated.c ↔
  y = 2 * (x + 3)^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2924_292432


namespace NUMINAMATH_CALUDE_triangle_3_4_5_l2924_292423

/-- A triangle can be formed from three line segments if the sum of the lengths of any two sides is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the line segments 3, 4, and 5 can form a triangle. -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_5_l2924_292423


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l2924_292475

/-- The set M of solutions to the quadratic equation 2x^2 - 3x - 2 = 0 -/
def M : Set ℝ := {x | 2 * x^2 - 3 * x - 2 = 0}

/-- The set N of solutions to the linear equation ax = 1 -/
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

/-- Theorem stating that if N is a subset of M, then a must be 0, -2, or 1/2 -/
theorem subset_implies_a_values (a : ℝ) (h : N a ⊆ M) : 
  a = 0 ∨ a = -2 ∨ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l2924_292475


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2924_292419

/-- A rhombus with side length 40 units and shorter diagonal 56 units has a longer diagonal of length 24√17 units. -/
theorem rhombus_longer_diagonal (s : ℝ) (d₁ : ℝ) (d₂ : ℝ) 
    (h₁ : s = 40) 
    (h₂ : d₁ = 56) 
    (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : 
  d₂ = 24 * Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2924_292419


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2924_292416

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 2 = 3) : 
  a 3 + a 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2924_292416


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l2924_292433

/-- The function f(x) = a^(-x-2) + 4 always passes through the point (-2, 5) for a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(-x-2) + 4
  f (-2) = 5 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l2924_292433


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2924_292471

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2924_292471


namespace NUMINAMATH_CALUDE_five_ps_high_gpa_l2924_292430

/-- Represents the number of applicants satisfying various criteria in a law school application process. -/
structure Applicants where
  total : ℕ
  political_science : ℕ
  high_gpa : ℕ
  not_ps_low_gpa : ℕ

/-- Calculates the number of applicants who majored in political science and had a GPA higher than 3.0. -/
def political_science_and_high_gpa (a : Applicants) : ℕ :=
  a.high_gpa - (a.total - a.political_science - a.not_ps_low_gpa)

/-- Theorem stating that for the given applicant data, 5 applicants majored in political science and had a GPA higher than 3.0. -/
theorem five_ps_high_gpa (a : Applicants) 
    (h_total : a.total = 40)
    (h_ps : a.political_science = 15)
    (h_high_gpa : a.high_gpa = 20)
    (h_not_ps_low_gpa : a.not_ps_low_gpa = 10) :
    political_science_and_high_gpa a = 5 := by
  sorry

#eval political_science_and_high_gpa ⟨40, 15, 20, 10⟩

end NUMINAMATH_CALUDE_five_ps_high_gpa_l2924_292430


namespace NUMINAMATH_CALUDE_min_value_w_l2924_292437

theorem min_value_w (x y : ℝ) : 
  ∃ (w_min : ℝ), w_min = 20.25 ∧ ∀ (w : ℝ), w = 3*x^2 + 3*y^2 + 9*x - 6*y + 27 → w ≥ w_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_w_l2924_292437


namespace NUMINAMATH_CALUDE_gcd_diff_is_square_l2924_292401

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x (Nat.gcd y z)) * (y - x) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_diff_is_square_l2924_292401


namespace NUMINAMATH_CALUDE_complex_multiplication_l2924_292400

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The complex number 3+2i -/
def z : ℂ := 3 + 2 * i

theorem complex_multiplication :
  z * i = -2 + 3 * i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2924_292400


namespace NUMINAMATH_CALUDE_rectangle_length_is_16_l2924_292414

/-- Proves that the length of a rectangle is 16 cm given specific conditions --/
theorem rectangle_length_is_16 (b : ℝ) (c : ℝ) :
  b = 14 →
  c = 23.56 →
  ∃ (l : ℝ), l = 16 ∧ 
    2 * (l + b) = 4 * (c / π) ∧
    c / π = (2 * c) / (2 * π) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_is_16_l2924_292414


namespace NUMINAMATH_CALUDE_min_hat_flips_min_hat_flips_1000_l2924_292473

theorem min_hat_flips (n : ℕ) (h : n = 1000) : ℕ :=
  let elf_count := n
  let initial_red_count : ℕ := n - 1
  let initial_blue_count : ℕ := 1
  let final_red_count : ℕ := 1
  let final_blue_count : ℕ := n - 1
  let min_flips := initial_red_count - final_red_count
  min_flips

/-- The minimum number of hat flips required for 1000 elves to satisfy the conditions is 998. -/
theorem min_hat_flips_1000 : min_hat_flips 1000 (by rfl) = 998 := by
  sorry

end NUMINAMATH_CALUDE_min_hat_flips_min_hat_flips_1000_l2924_292473


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l2924_292462

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = x * (1 - x)) :
  ∀ x < 0, f x = x * (1 + x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l2924_292462


namespace NUMINAMATH_CALUDE_PropB_implies_PropA_PropA_not_implies_PropB_A_necessary_not_sufficient_for_B_l2924_292438

/-- Proposition A: x ≠ 2 or y ≠ 3 -/
def PropA (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 3

/-- Proposition B: x + y ≠ 5 -/
def PropB (x y : ℝ) : Prop := x + y ≠ 5

/-- Proposition B implies Proposition A -/
theorem PropB_implies_PropA : ∀ x y : ℝ, PropB x y → PropA x y := by sorry

/-- Proposition A does not imply Proposition B -/
theorem PropA_not_implies_PropB : ¬(∀ x y : ℝ, PropA x y → PropB x y) := by sorry

/-- A is a necessary but not sufficient condition for B -/
theorem A_necessary_not_sufficient_for_B : 
  (∀ x y : ℝ, PropB x y → PropA x y) ∧ ¬(∀ x y : ℝ, PropA x y → PropB x y) := by sorry

end NUMINAMATH_CALUDE_PropB_implies_PropA_PropA_not_implies_PropB_A_necessary_not_sufficient_for_B_l2924_292438


namespace NUMINAMATH_CALUDE_unique_solution_in_p_arithmetic_l2924_292435

-- Define p-arithmetic structure
structure PArithmetic (p : ℕ) where
  carrier : Type
  add : carrier → carrier → carrier
  mul : carrier → carrier → carrier
  zero : carrier
  one : carrier
  -- Add necessary axioms for p-arithmetic

-- Define the theorem
theorem unique_solution_in_p_arithmetic {p : ℕ} (P : PArithmetic p) :
  ∀ (a b : P.carrier), a ≠ P.zero → ∃! x : P.carrier, P.mul a x = b :=
sorry

end NUMINAMATH_CALUDE_unique_solution_in_p_arithmetic_l2924_292435


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_ratio_l2924_292470

theorem rectangle_area_diagonal_ratio (length width diagonal : ℝ) (k : ℝ) : 
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 4 / 3 →
  diagonal ^ 2 = length ^ 2 + width ^ 2 →
  length * width = k * diagonal ^ 2 →
  k = 12 / 25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_ratio_l2924_292470


namespace NUMINAMATH_CALUDE_least_value_quadratic_inequality_l2924_292408

theorem least_value_quadratic_inequality :
  (∀ b : ℝ, b < 4 → -b^2 + 9*b - 20 < 0) ∧
  (-4^2 + 9*4 - 20 = 0) ∧
  (∀ b : ℝ, b > 4 → -b^2 + 9*b - 20 ≤ -b^2 + 9*4 - 20) :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_inequality_l2924_292408


namespace NUMINAMATH_CALUDE_recorder_price_problem_l2924_292443

theorem recorder_price_problem :
  ∀ (a b : ℕ),
    a < 5 ∧ b < 10 →  -- Ensure the old price is less than 50
    (10 * b + a : ℚ) = (10 * a + b : ℚ) * (6/5) →  -- 20% increase and digit swap
    10 * b + a = 54 :=
by sorry

end NUMINAMATH_CALUDE_recorder_price_problem_l2924_292443


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2924_292459

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (((a : ℂ) - Complex.I) / (1 + Complex.I)).re = 0 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2924_292459


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2924_292469

theorem intersection_of_lines : 
  ∃! (x y : ℚ), 2 * y = 3 * x ∧ 3 * y + 1 = -6 * x ∧ x = -2/21 ∧ y = -1/7 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2924_292469


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2924_292406

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2924_292406


namespace NUMINAMATH_CALUDE_original_stones_count_l2924_292426

/-- The number of stones sent away to the Geological Museum in London. -/
def stones_sent_away : ℕ := 63

/-- The number of stones kept in the collection. -/
def stones_kept : ℕ := 15

/-- The original number of stones in the collection. -/
def original_stones : ℕ := stones_sent_away + stones_kept

/-- Theorem stating that the original number of stones in the collection is 78. -/
theorem original_stones_count : original_stones = 78 := by
  sorry

end NUMINAMATH_CALUDE_original_stones_count_l2924_292426


namespace NUMINAMATH_CALUDE_light_ray_reflection_l2924_292477

/-- Represents a direction vector in 3D space -/
structure DirectionVector where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a trirectangular corner -/
structure TrirectangularCorner where

/-- Reflects a direction vector off a plane perpendicular to the x-axis -/
def reflectX (v : DirectionVector) : DirectionVector :=
  { x := -v.x, y := v.y, z := v.z }

/-- Reflects a direction vector off a plane perpendicular to the y-axis -/
def reflectY (v : DirectionVector) : DirectionVector :=
  { x := v.x, y := -v.y, z := v.z }

/-- Reflects a direction vector off a plane perpendicular to the z-axis -/
def reflectZ (v : DirectionVector) : DirectionVector :=
  { x := v.x, y := v.y, z := -v.z }

/-- 
  Theorem: A light ray reflecting off all three faces of a trirectangular corner
  will change its direction to the opposite of its initial direction.
-/
theorem light_ray_reflection 
  (corner : TrirectangularCorner) 
  (initial_direction : DirectionVector) :
  reflectX (reflectY (reflectZ initial_direction)) = 
  { x := -initial_direction.x, 
    y := -initial_direction.y, 
    z := -initial_direction.z } := by
  sorry


end NUMINAMATH_CALUDE_light_ray_reflection_l2924_292477


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l2924_292405

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of two runners being in a picture -/
def probabilityInPicture (jenny : Runner) (jack : Runner) : ℚ :=
  -- Define the probability calculation here
  23 / 60

/-- Theorem stating the probability of Jenny and Jack being in the picture -/
theorem runners_in_picture_probability :
  let jenny : Runner := { lapTime := 75, direction := true }
  let jack : Runner := { lapTime := 70, direction := false }
  let pictureTime : ℝ := 15 * 60  -- 15 minutes in seconds
  let pictureDuration : ℝ := 60  -- 1 minute in seconds
  let pictureTrackCoverage : ℝ := 1 / 3
  probabilityInPicture jenny jack = 23 / 60 := by
  sorry

#eval probabilityInPicture { lapTime := 75, direction := true } { lapTime := 70, direction := false }

end NUMINAMATH_CALUDE_runners_in_picture_probability_l2924_292405


namespace NUMINAMATH_CALUDE_thursday_monday_difference_l2924_292480

/-- Represents the number of bonnets made on each day of the week --/
structure BonnetProduction where
  monday : ℕ
  tuesday_wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the difference between Thursday and Monday bonnet production --/
theorem thursday_monday_difference (bp : BonnetProduction) : 
  bp.monday = 10 →
  bp.tuesday_wednesday = 2 * bp.monday →
  bp.friday = bp.thursday - 5 →
  (bp.monday + bp.tuesday_wednesday + bp.thursday + bp.friday) / 5 = 11 →
  bp.thursday - bp.monday = 5 := by
  sorry

end NUMINAMATH_CALUDE_thursday_monday_difference_l2924_292480


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l2924_292448

/-- Two-digit integer -/
def TwoDigitInt (z : ℕ) : Prop := 10 ≤ z ∧ z ≤ 99

/-- Reverse digits of a two-digit number -/
def reverseDigits (x : ℕ) : ℕ := 10 * (x % 10) + (x / 10)

theorem two_digit_reverse_sum (x y n : ℕ) : 
  TwoDigitInt x → TwoDigitInt y → y = reverseDigits x → x^2 - y^2 = n^2 → x + y + n = 154 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l2924_292448


namespace NUMINAMATH_CALUDE_division_problem_l2924_292454

theorem division_problem (n : ℕ) : n = 867 → n / 37 = 23 ∧ n % 37 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2924_292454


namespace NUMINAMATH_CALUDE_nine_to_150_mod_50_l2924_292488

theorem nine_to_150_mod_50 : 9^150 % 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_nine_to_150_mod_50_l2924_292488


namespace NUMINAMATH_CALUDE_function_properties_l2924_292457

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem function_properties (a b : ℝ) :
  (∃ y, f a b 1 = y ∧ x + y - 3 = 0) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≤ 8) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≥ -4) ∧
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, ∃ y ∈ Set.Ioo (-1 : ℝ) 1, x < y ∧ f a b x > f a b y) →
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2924_292457


namespace NUMINAMATH_CALUDE_square_difference_l2924_292460

theorem square_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2924_292460


namespace NUMINAMATH_CALUDE_flagpole_break_height_l2924_292463

/-- Given a flagpole of height 8 meters that breaks and touches the ground 3 meters from its base,
    the height of the break point is √3 meters. -/
theorem flagpole_break_height :
  ∀ (h x : ℝ),
  h = 8 →  -- Original height of flagpole
  x^2 + 3^2 = (h - x)^2 →  -- Pythagorean theorem application
  x = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l2924_292463


namespace NUMINAMATH_CALUDE_tournament_participants_l2924_292492

theorem tournament_participants :
  ∃ n : ℕ,
    n > 0 ∧
    (n - 2) * (n - 3) / 2 + 7 = 62 ∧
    n = 13 :=
by sorry

end NUMINAMATH_CALUDE_tournament_participants_l2924_292492


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l2924_292467

/-- The number of grandchildren -/
def n : ℕ := 12

/-- The probability of a child being male (or female) -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of males and females
    given n independently determined genders with probability p of being male -/
def unequal_probability (n : ℕ) (p : ℚ) : ℚ :=
  1 - (n.choose (n/2) : ℚ) * p^(n/2) * (1-p)^(n/2)

/-- The theorem to be proved -/
theorem unequal_gender_probability :
  unequal_probability n p = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l2924_292467
