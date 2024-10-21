import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_disjoint_routes_l517_51715

/-- Represents an intersection in the city district -/
structure Intersection :=
  (id : ℕ)

/-- Represents the city district -/
structure CityDistrict where
  intersections : Finset Intersection
  routes : Intersection → Intersection → Set (List Intersection)
  at_least_three_intersections : 3 ≤ intersections.card
  route_not_through : ∀ (A B C : Intersection), 
    ∃ (route : List Intersection), route ∈ routes A B ∧ C ∉ route

/-- Two routes are disjoint if they don't share any intersections except the start and end -/
def disjoint_routes (district : CityDistrict) (A B : Intersection) 
  (route1 route2 : List Intersection) : Prop :=
  route1 ∈ district.routes A B ∧ 
  route2 ∈ district.routes A B ∧ 
  (∀ I : Intersection, I ≠ A ∧ I ≠ B → (I ∈ route1 → I ∉ route2) ∧ (I ∈ route2 → I ∉ route1))

/-- The main theorem to be proved -/
theorem two_disjoint_routes (district : CityDistrict) :
  ∀ (A B : Intersection), A ∈ district.intersections → B ∈ district.intersections → A ≠ B →
  ∃ (route1 route2 : List Intersection), disjoint_routes district A B route1 route2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_disjoint_routes_l517_51715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_price_l517_51798

/-- Represents the selling price calculation for cloth. -/
structure ClothSale where
  /-- Total length of cloth sold in meters -/
  length : ℕ
  /-- Cost price per meter in rupees -/
  costPrice : ℕ
  /-- Profit per meter in rupees -/
  profitPerMeter : ℕ

/-- Calculates the total selling price of cloth. -/
def totalSellingPrice (sale : ClothSale) : ℕ :=
  sale.length * (sale.costPrice + sale.profitPerMeter)

/-- Theorem stating that the total selling price for the given conditions is 4500 rupees. -/
theorem cloth_sale_price (sale : ClothSale) 
  (h1 : sale.length = 45)
  (h2 : sale.costPrice = 86)
  (h3 : sale.profitPerMeter = 14) :
  totalSellingPrice sale = 4500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_price_l517_51798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snow_volume_l517_51702

/-- The total volume of snow to shovel from two portions of sidewalk -/
theorem total_snow_volume
  (main_length main_width main_depth square_side square_depth : ℝ)
  (h1 : main_length = 30)
  (h2 : main_width = 3)
  (h3 : main_depth = 0.5)
  (h4 : square_side = 4)
  (h5 : square_depth = 1) :
  main_length * main_width * main_depth + square_side * square_side * square_depth = 61 := by
  -- Substitute the given values
  rw [h1, h2, h3, h4, h5]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

#check total_snow_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snow_volume_l517_51702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l517_51728

-- Define the slope of a line given its equation in the form ax + by = c
def line_slope (a b : ℚ) : ℚ := -a / b

-- Define what it means for two lines to be parallel
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℚ) : Prop :=
  line_slope a₁ b₁ = line_slope a₂ b₂

-- Theorem statement
theorem parallel_line_slope :
  ∀ (a b c : ℚ), parallel 3 (-6) 12 a b c → line_slope a b = 1/2 :=
by
  intros a b c h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l517_51728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_percentage_l517_51786

theorem recreation_spending_percentage
  (last_week_wages : ℝ)
  (last_week_recreation_percentage : ℝ)
  (wage_decrease_percentage : ℝ)
  (this_week_recreation_ratio : ℝ)
  (h1 : last_week_recreation_percentage = 0.30)
  (h2 : wage_decrease_percentage = 0.25)
  (h3 : this_week_recreation_ratio = 0.50) :
  let this_week_wages := last_week_wages * (1 - wage_decrease_percentage)
  let this_week_recreation := last_week_wages * last_week_recreation_percentage * this_week_recreation_ratio
  let this_week_recreation_percentage := this_week_recreation / this_week_wages
  this_week_recreation_percentage = 0.20 := by
  -- Proof steps would go here
  sorry

#check recreation_spending_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_percentage_l517_51786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_prime_power_product_l517_51704

theorem factors_of_prime_power_product (M : ℕ) :
  M = 2^4 * 3^3 * 5^2 * 7^1 →
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_prime_power_product_l517_51704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l517_51731

def sequenceSum (n : ℕ) : ℕ := n.factorial + n

def sum_sequence : ℕ := (List.range 12).map (λ i => sequenceSum (i + 1)) |>.sum

theorem units_digit_of_sum_sequence : sum_sequence % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_sequence_l517_51731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_5n_l517_51748

theorem divisors_of_5n (n : ℕ) 
  (h : (Nat.divisors (36 * n^2)).card = 51) : 
  (Nat.divisors (5 * n)).card = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_5n_l517_51748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l517_51752

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l517_51752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l517_51760

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x * Real.exp x else x^2 - 2*x + 1/2

theorem f_properties :
  (∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f (-2) ∧ deriv f (-2) = -2 / Real.exp 2) ∧
  (∀ x, f x ≥ -2 / Real.exp 1) ∧
  (∃ (min_point : ℝ), f min_point = -2 / Real.exp 1) ∧
  (∀ x y, x < y ∧ y ≤ -1 → f y < f x) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y ≤ 1 → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l517_51760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_20_value_l517_51743

/-- Sequence c_n defined recursively -/
def c : ℕ → ℕ
  | 0 => 3  -- Define c₁
  | 1 => 9  -- Define c₂
  | (n + 2) => c (n + 1) * c n

/-- Theorem stating that the 20th term of sequence c_n equals 3^10946 -/
theorem c_20_value : c 19 = 3^10946 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_20_value_l517_51743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_angle_is_60_particles_meet_l517_51783

/-- The time (in seconds) it takes for the particles to meet -/
def meeting_time : ℝ := 10

/-- The initial distance between the particles in meters -/
def initial_distance : ℝ := 295

/-- The speed of the first particle in meters per second -/
def speed_first_particle : ℝ := 15

/-- The distance covered by the second particle in the nth second -/
def distance_second_particle (n : ℕ) : ℝ := 3 * (n : ℝ) - 2

/-- The angle the second hand of a clock moves per second in degrees -/
def second_hand_speed : ℝ := 6

/-- The theorem stating that the angle the second hand moves during the meeting time is 60 degrees -/
theorem second_hand_angle_is_60 : 
  meeting_time * second_hand_speed = 60 := by
  -- Proof goes here
  sorry

/-- The total distance traveled by both particles equals the initial distance -/
theorem particles_meet :
  speed_first_particle * meeting_time + 
  (meeting_time * (3 * meeting_time - 1)) / 2 = initial_distance := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_hand_angle_is_60_particles_meet_l517_51783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_2001_f_property_1_f_property_2_l517_51795

-- Define the function f as noncomputable due to its dependence on real operations
noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 3 then 1 - |x - 2|
  else if x > 0 then
    let k := Real.log x / Real.log 3
    let n := Int.floor k
    let y := x / (3 ^ n)
    3^n * (1 - |y - 2|)
  else 0  -- Definition for non-positive x (not specified in the original problem)

-- State the theorem
theorem smallest_x_equals_f_2001 :
  ∀ x > 0, f x = f 2001 → x ≥ 429 ∧ f 429 = f 2001 := by
  sorry

-- Additional properties of f that might be useful for the proof
theorem f_property_1 : ∀ x > 0, f (3 * x) = 3 * f x := by
  sorry

theorem f_property_2 : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x = 1 - |x - 2| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_2001_f_property_1_f_property_2_l517_51795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l517_51700

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x - 1 / 2

theorem f_monotone_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (-Real.pi / 3 + ↑k * Real.pi) (Real.pi / 6 + ↑k * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l517_51700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_relation_l517_51730

/-- Given a function g and its inverse f⁻¹, prove that 7a + 7b = 5 -/
theorem inverse_function_relation (a b : ℝ) :
  (∀ x, (7 * x - 6) = ((λ y ↦ a * y + b)⁻¹ x) - 2) →
  7 * a + 7 * b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_relation_l517_51730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_runs_by_running_approx_56_52_l517_51703

/-- Calculates the percentage of runs made by running between the wickets -/
noncomputable def percentage_runs_by_running (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℝ :=
  let boundary_runs := 4 * boundaries
  let six_runs := 6 * sixes
  let runs_without_running := boundary_runs + six_runs
  let runs_by_running := total_runs - runs_without_running
  (runs_by_running : ℝ) / (total_runs : ℝ) * 100

/-- The percentage of runs made by running between the wickets is approximately 56.52% -/
theorem percentage_runs_by_running_approx_56_52 :
  |percentage_runs_by_running 138 12 2 - 56.52| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_runs_by_running_approx_56_52_l517_51703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l517_51789

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * (2:ℝ)^x - (2:ℝ)^(-x))

-- State the theorem
theorem even_function_implies_a_equals_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l517_51789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_theorem_l517_51709

open BigOperators Finset

variable (n m : ℕ)

def female_students : ℕ := 5
def male_students : ℕ := 4

def select_two_each : ℕ := (Nat.choose male_students 2) * (Nat.choose female_students 2)

def select_at_least_one_each : ℕ := 
  Nat.choose (female_students + male_students) 4 - Nat.choose female_students 4 - Nat.choose male_students 4

def select_with_restriction (a b : ℕ) : ℕ := 
  select_at_least_one_each - (Nat.choose (female_students + male_students - 2) 2)

theorem selection_theorem : 
  select_two_each = 60 ∧ 
  select_at_least_one_each = 120 ∧ 
  select_with_restriction 1 1 = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_theorem_l517_51709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l517_51785

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the area of triangle ODE
def triangle_area (a b : ℝ) : ℝ := a * b

-- Define the focal length of the hyperbola
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

theorem min_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola a b a b ∧ hyperbola a b a (-b) ∧ triangle_area a b = 8 →
  ∃ (min_fl : ℝ), min_fl = 8 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    hyperbola a' b' a' b' ∧ hyperbola a' b' a' (-b') ∧ triangle_area a' b' = 8 →
    focal_length a' b' ≥ min_fl :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l517_51785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_squarish_modified_exactly_two_squarish_modified_l517_51771

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns the first three digits of a five-digit number --/
def first_three_digits (n : ℕ) : ℕ :=
  n / 100

/-- A function that returns the last two digits of a five-digit number --/
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

/-- A function that checks if a number is squarish-modified --/
def is_squarish_modified (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n ≤ 99999) ∧  -- five-digit number
  (∀ d, d ∈ Nat.digits 10 n → d ≠ 0) ∧  -- no digit is zero
  is_perfect_square n ∧  -- perfect square
  is_perfect_square (first_three_digits n) ∧  -- first three digits are perfect square
  is_perfect_square (last_two_digits n)  -- last two digits are perfect square

/-- Theorem: There exists at least one squarish-modified number --/
theorem exists_squarish_modified : ∃ n : ℕ, is_squarish_modified n := by
  sorry

/-- Theorem: There are exactly two squarish-modified numbers --/
theorem exactly_two_squarish_modified :
  ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧
  is_squarish_modified n₁ ∧ is_squarish_modified n₂ ∧
  ∀ n : ℕ, is_squarish_modified n → (n = n₁ ∨ n = n₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_squarish_modified_exactly_two_squarish_modified_l517_51771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonprime_term_l517_51766

/-- A sequence defined by x_{n+1} = ax_n + b -/
def sequenceX (a b x₀ : ℕ+) : ℕ → ℕ
  | 0 => x₀
  | n + 1 => a * sequenceX a b x₀ n + b

/-- Theorem: In the sequence defined by x_{n+1} = ax_n + b, 
    there exists a non-prime term for some i ≥ 1 -/
theorem exists_nonprime_term (a b x₀ : ℕ+) :
  ∃ i : ℕ, i ≥ 1 ∧ ¬ Nat.Prime (sequenceX a b x₀ i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonprime_term_l517_51766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AC_equation_max_area_OPQ_l517_51729

-- Define the circle B
def circle_B (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the lines l1 and l2 passing through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define perpendicular lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the dot product of vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Define the length of a vector
noncomputable def vector_length (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem line_AC_equation (x y : ℝ) (m1 m2 : ℝ) (xA yA xB yB xC yC : ℝ) :
  circle_B xB yB →
  line_through_origin m1 xA yA →
  line_through_origin m2 xC yC →
  perpendicular m1 m2 →
  dot_product (xA - xB) (yA - yB) (xC - xB) (yC - yB) = 0 →
  vector_length (xA - xB) (yA - yB) = vector_length (xC - xB) (yC - yB) →
  x + y - 1 = 0 := by sorry

theorem max_area_OPQ (xP yP xQ yQ : ℝ) (m1 m2 : ℝ) :
  circle_B xP yP →
  circle_B xQ yQ →
  line_through_origin m1 xP yP →
  line_through_origin m2 xQ yQ →
  perpendicular m1 m2 →
  ∀ x y, circle_B x y → line_through_origin m1 x y → line_through_origin m2 x y →
    vector_length x y * vector_length xP yP * vector_length xQ yQ / 2 ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AC_equation_max_area_OPQ_l517_51729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_sum_neq_edge_sum_l517_51774

/-- Represents a cube with natural numbers at its vertices -/
structure NumberedCube where
  vertices : Fin 8 → ℕ
  distinct : ∀ i j, i ≠ j → vertices i ≠ vertices j

/-- The sum of the numbers at the vertices of the cube -/
def vertexSum (cube : NumberedCube) : ℕ :=
  (Finset.univ.sum fun i => cube.vertices i)

/-- The sum of the GCDs of the numbers at the ends of each edge of the cube -/
noncomputable def edgeSum (cube : NumberedCube) : ℕ :=
  let edges : List (Fin 8 × Fin 8) := sorry -- List of all 12 edges of the cube
  (List.sum (edges.map fun (i, j) => Nat.gcd (cube.vertices i) (cube.vertices j)))

/-- Theorem stating that the vertex sum cannot equal the edge sum for any numbered cube -/
theorem vertex_sum_neq_edge_sum (cube : NumberedCube) : vertexSum cube ≠ edgeSum cube := by
  sorry

#check vertex_sum_neq_edge_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_sum_neq_edge_sum_l517_51774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_correct_l517_51797

/-- A rhombus with given diagonal lengths and a point on one side -/
structure RhombusWithPoint where
  -- The length of diagonal AC
  ac : ℝ
  -- The length of diagonal BD
  bd : ℝ
  -- The angle DAN in radians
  angle_dan : ℝ
  -- Assumption that AC = 20
  ac_eq : ac = 20
  -- Assumption that BD = 24
  bd_eq : bd = 24
  -- Assumption that ∠DAN = 30°
  angle_dan_eq : angle_dan = π / 6

/-- The minimum distance between perpendicular feet -/
noncomputable def min_distance (r : RhombusWithPoint) : ℝ :=
  2 * Real.sqrt 61

theorem min_distance_is_correct (r : RhombusWithPoint) :
  min_distance r = 2 * Real.sqrt 61 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_correct_l517_51797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharp_triple_40_l517_51772

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem sharp_triple_40 : Int.floor (sharp (Int.floor (sharp (Int.floor (sharp 40))))) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharp_triple_40_l517_51772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_road_trip_l517_51721

theorem rick_road_trip (leg1 leg2 leg3 leg4 : ℕ) : 
  leg2 = 2 * leg1 →
  leg3 = 40 →
  leg3 * 2 = leg1 →
  leg4 = 2 * (leg1 + leg2 + leg3) →
  leg1 + leg2 + leg3 + leg4 = 560 := by
  sorry

#check rick_road_trip

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_road_trip_l517_51721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_periodicity_l517_51790

noncomputable def x_sequence (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => let xₙ := x_sequence x₀ n
              if 2 * xₙ < 1 then 2 * xₙ else 2 * xₙ - 1

theorem x_sequence_periodicity : 
  ∃ (s : Finset ℝ), s.card = 31 ∧ 
  (∀ x₀ ∈ s, 0 ≤ x₀ ∧ x₀ < 1 ∧ x_sequence x₀ 0 = x_sequence x₀ 5) ∧
  (∀ x₀, 0 ≤ x₀ → x₀ < 1 → x_sequence x₀ 0 = x_sequence x₀ 5 → x₀ ∈ s) := by
  sorry

#check x_sequence_periodicity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_periodicity_l517_51790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l517_51792

theorem shaded_area_calculation (r : ℝ) (θ : ℝ) (h1 : r = 3) (h2 : θ = π/4) : 
  r^2 * (θ - Real.sin θ) = 9 * (π - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l517_51792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_large_k_l517_51770

theorem no_solutions_for_large_k : ∃ k₀ : ℝ, k₀ = 4028 ∧ 
  ∀ k a b n : ℕ, k ≥ k₀ → 
  a^(2*n) + b^(4*n) + 2013 ≠ k * a^n * b^(2*n) :=
by
  -- We'll use natural numbers (ℕ) instead of reals (ℝ) for a, b, n, and k
  -- This eliminates the need for IsInt and positivity checks
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_large_k_l517_51770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_when_a_is_e_f_nonnegative_iff_a_in_zero_one_l517_51732

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - a

theorem f_minimum_when_a_is_e :
  ∃ (x : ℝ), f (Real.exp 1) x = -(Real.exp 1) ∧ ∀ y : ℝ, f (Real.exp 1) y ≥ -(Real.exp 1) :=
sorry

theorem f_nonnegative_iff_a_in_zero_one :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_when_a_is_e_f_nonnegative_iff_a_in_zero_one_l517_51732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l517_51799

/-- The eccentricity of a hyperbola whose focus coincides with the focus of a parabola -/
theorem hyperbola_eccentricity (a : ℝ) : 
  (∃ x y : ℝ, y^2 = 8*x ∧ x^2/a^2 - y^2 = 1 ∧ x = 2 ∧ y = 0) →
  (Real.sqrt (a^2 + 1))/a = 2*Real.sqrt 3/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l517_51799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l517_51769

noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x - Real.pi / 6) + 1

theorem function_properties (A ω : ℝ) (hA : A > 0) (hω : ω > 0)
  (hmin : ∀ x, f A ω x ≥ -1)
  (hperiod : ∀ x, f A ω (x + Real.pi / ω) = f A ω x) :
  A = 2 ∧ ω = 2 ∧
  ∀ α, α ∈ Set.Ioo 0 (Real.pi / 2) → f A ω (α / 2) = 2 → α = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l517_51769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l517_51710

theorem triangle_ABC_properties (B C : ℝ) (b c : ℝ) :
  Real.cos (2 * B) = -(1 / 2) →
  c = 8 →
  b = 7 →
  Real.sin C = (4 * Real.sqrt 3) / 7 ∧
  (C > Real.pi / 2 → b + c + Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos C)) = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l517_51710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l517_51711

/-- Represents a digit in base 7 -/
def Digit7 := { n : ℕ // n < 7 ∧ n ≠ 0 }

/-- Converts a two-digit number in base 7 to its decimal representation -/
def toDecimal (a b : Digit7) : ℕ :=
  7 * a.val + b.val

/-- Converts a three-digit number in base 7 to its decimal representation -/
def toDecimal3 (a b c : Digit7) : ℕ :=
  49 * a.val + 7 * b.val + c.val

/-- The theorem statement -/
theorem unique_solution (A B C : Digit7) :
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  toDecimal A B + C.val = 7 * C.val →
  toDecimal A B + toDecimal B A = toDecimal A C →
  toDecimal3 A B C = 516 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l517_51711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_interval_f_not_strictly_increasing_outside_l517_51791

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- State the theorem
theorem f_strictly_increasing_interval :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < exp 1 → f x₁ < f x₂ :=
by
  sorry

-- State the theorem for the complement
theorem f_not_strictly_increasing_outside :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ exp 1 ≤ x₁ → f x₁ ≥ f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_interval_f_not_strictly_increasing_outside_l517_51791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l517_51761

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    and a circle with diameter equal to the distance between the foci,
    prove that if the angle formed by a point on the hyperbola, the origin,
    and the right focus is equal to the angle formed by a point on the asymptote,
    the origin, and the point where the circle intersects the positive y-axis,
    then the eccentricity of the hyperbola is (√5 + 1)/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y ↦ x^2/a^2 - y^2/b^2 = 1
  let c := Real.sqrt (a^2 + b^2)
  let circle : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 = c^2
  let asymptote : ℝ → ℝ → Prop := λ x y ↦ y = (b/a) * x
  ∃ (P Q : ℝ × ℝ) (B : ℝ),
    C P.1 P.2 ∧ circle P.1 P.2 ∧
    asymptote Q.1 Q.2 ∧ circle Q.1 Q.2 ∧
    circle 0 B ∧
    (P.2 / P.1 = Q.1 / Q.2) →
    Real.sqrt (1 + b^2/a^2) = (Real.sqrt 5 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l517_51761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_is_six_l517_51780

/-- A triangular prism with points on its lateral edges -/
structure TriangularPrism where
  volume : ℝ
  am_ratio : ℝ
  bn_ratio : ℝ
  ck_ratio : ℝ

/-- The maximum volume of a pyramid formed by points on the lateral edges and any point in the prism -/
noncomputable def max_pyramid_volume (prism : TriangularPrism) : ℝ := 
  (1/3) * min prism.am_ratio (min prism.bn_ratio prism.ck_ratio) * prism.volume

/-- Theorem stating the maximum volume of the pyramid MNKP -/
theorem max_pyramid_volume_is_six (prism : TriangularPrism) 
  (h_volume : prism.volume = 27)
  (h_am : prism.am_ratio = 2/3)
  (h_bn : prism.bn_ratio = 3/5)
  (h_ck : prism.ck_ratio = 4/7) :
  max_pyramid_volume prism = 6 := by
  sorry

#check max_pyramid_volume_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_is_six_l517_51780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l517_51744

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- The problem statement -/
theorem interest_problem (P : ℝ) : 
  simple_interest P 12 4 = (1/2) * compound_interest 6000 15 2 → 
  P = 2015.625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l517_51744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_chord_length_l517_51722

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ∀ x y : ℝ, y^2 = 4*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (1, 0)

/-- A chord of a parabola -/
structure Chord (p : Parabola) where
  start : ℝ × ℝ
  finish : ℝ × ℝ

/-- A chord is perpendicular to the axis of symmetry if its x-coordinates are equal -/
def is_perpendicular_to_axis {p : Parabola} (c : Chord p) : Prop :=
  c.start.1 = c.finish.1

/-- A chord passes through the focus if one of its endpoints is the focus -/
def passes_through_focus {p : Parabola} (c : Chord p) : Prop :=
  c.start = focus p ∨ c.finish = focus p

/-- The length of a chord -/
noncomputable def chord_length {p : Parabola} (c : Chord p) : ℝ :=
  Real.sqrt ((c.finish.1 - c.start.1)^2 + (c.finish.2 - c.start.2)^2)

/-- The main theorem: the length of the chord perpendicular to the axis of symmetry
    and passing through the focus is 4 units -/
theorem focus_chord_length (p : Parabola) (c : Chord p)
  (h1 : is_perpendicular_to_axis c)
  (h2 : passes_through_focus c) :
  chord_length c = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_chord_length_l517_51722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l517_51787

-- Define the circles and their properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the tangency relations
def internallyTangent (c1 c2 : Set (ℝ × ℝ)) : Prop := sorry
def externallyTangent (c1 c2 : Set (ℝ × ℝ)) : Prop := sorry
def tangentToLine (c : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop := sorry

-- Define the theorem
theorem circle_tangency_theorem :
  ∀ (C D E : Set (ℝ × ℝ)) (A : ℝ × ℝ) (AB : Set (ℝ × ℝ)),
    (∃ (center_C center_D center_E : ℝ × ℝ) (r_C r_D r_E : ℝ),
      C = Circle center_C r_C ∧
      D = Circle center_D r_D ∧
      E = Circle center_E r_E ∧
      r_C = 2 ∧
      r_D = 3 * r_E ∧
      internallyTangent D C ∧
      A ∈ C ∩ D ∧
      internallyTangent E C ∧
      externallyTangent E D ∧
      tangentToLine E AB) →
    ∃ (m n : ℕ), r_D = Real.sqrt (m : ℝ) - n ∧ m = 240 ∧ n = 14 :=
by
  sorry

#check circle_tangency_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l517_51787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_approx_631_51_l517_51718

/-- Represents the cost and sale information for an article -/
structure Article where
  loss_or_gain : Float
  price_change : Float
  new_loss_or_gain : Float

/-- Calculates the cost price of an article based on its sale information -/
def calculate_cost_price (article : Article) : Float :=
  article.price_change / (article.new_loss_or_gain - article.loss_or_gain)

/-- The three articles from the problem -/
def article1 : Article := { loss_or_gain := -0.15, price_change := 72.50, new_loss_or_gain := 0.125 }
def article2 : Article := { loss_or_gain := 0.20, price_change := -45.30, new_loss_or_gain := -0.05 }
def article3 : Article := { loss_or_gain := -0.08, price_change := 33.60, new_loss_or_gain := 0.10 }

/-- Theorem stating that the total cost price of the three articles is approximately $631.51 -/
theorem total_cost_price_approx_631_51 :
  Float.abs ((calculate_cost_price article1) + (calculate_cost_price article2) + (calculate_cost_price article3) - 631.51) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_approx_631_51_l517_51718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_m_l517_51705

theorem perfect_square_m (m n k : ℕ) 
  (h : (1 : ℝ) + m + n * Real.sqrt 3 = (2 + Real.sqrt 3) ^ (2 * k + 1)) : 
  ∃ q : ℕ, m = q ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_m_l517_51705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l517_51794

/-- Represents the efficiency of a worker -/
inductive Efficiency
| Normal
| Double

/-- Calculates the effective number of workers based on their efficiency -/
def effectiveWorkers (workers : ℕ) (efficientWorkers : ℕ) : ℚ :=
  (workers - efficientWorkers : ℚ) + 2 * efficientWorkers

/-- Theorem stating the time taken to build the wall with new workforce -/
theorem wall_building_time 
  (initialWorkers initialEfficientWorkers initialDays : ℕ)
  (newWorkers newEfficientWorkers : ℕ) :
  initialWorkers > 0 → initialDays > 0 →
  initialWorkers = 18 → initialEfficientWorkers = 3 → initialDays = 3 →
  newWorkers = 30 → newEfficientWorkers = 5 →
  (effectiveWorkers newWorkers newEfficientWorkers * 
    (effectiveWorkers initialWorkers initialEfficientWorkers * initialDays) / 
    effectiveWorkers newWorkers newEfficientWorkers) = 9/5 := by
  sorry

#check wall_building_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l517_51794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_equal_or_supplementary_l517_51776

-- Define points O, A, B, O₁, A₁, B₁ in a plane
variable (O A B O₁ A₁ B₁ : EuclideanPlane)

-- Define the angles as functions
def angle_AOB (O A B : EuclideanPlane) : ℝ := sorry
def angle_A₁O₁B₁ (O₁ A₁ B₁ : EuclideanPlane) : ℝ := sorry

-- Define the parallel relationships as functions
def OA_parallel_O₁A₁ (O A O₁ A₁ : EuclideanPlane) : Prop := sorry
def OB_parallel_O₁B₁ (O B O₁ B₁ : EuclideanPlane) : Prop := sorry

-- State the theorem
theorem angles_equal_or_supplementary 
  (h1 : OA_parallel_O₁A₁ O A O₁ A₁) 
  (h2 : OB_parallel_O₁B₁ O B O₁ B₁) : 
  (angle_AOB O A B = angle_A₁O₁B₁ O₁ A₁ B₁) ∨ 
  (angle_AOB O A B + angle_A₁O₁B₁ O₁ A₁ B₁ = π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_equal_or_supplementary_l517_51776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_integer_proof_greatest_three_digit_integer_l517_51747

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def is_integer_division (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem greatest_three_digit_integer (n : ℕ) : Prop :=
  n = 987 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧
  is_integer_division (n^2) (sum_of_first_n n) ∧
  ¬(Nat.factorial n % (sum_of_first_n n)^2 = 0) ∧
  ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m > n →
    ¬(is_integer_division (m^2) (sum_of_first_n m) ∧
      ¬(Nat.factorial m % (sum_of_first_n m)^2 = 0))

-- Proof
theorem proof_greatest_three_digit_integer : ∃ n : ℕ, greatest_three_digit_integer n :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_integer_proof_greatest_three_digit_integer_l517_51747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_division_l517_51740

/-- Represents a partnership with two partners -/
structure Partnership where
  investment1 : ℝ
  investment2 : ℝ
  totalProfit : ℝ
  equalPortion : ℝ
  profitDifference : ℝ

/-- Calculates the profit share for a partner based on their investment ratio -/
noncomputable def profitShare (p : Partnership) (investment : ℝ) : ℝ :=
  (investment / (p.investment1 + p.investment2)) * (p.totalProfit - p.equalPortion)

theorem partnership_profit_division 
  (p : Partnership)
  (h1 : p.investment1 = 700)
  (h2 : p.investment2 = 300)
  (h3 : p.totalProfit = 3000)
  (h4 : p.profitDifference = 800)
  (h5 : p.equalPortion / 2 + profitShare p p.investment1 = 
        p.equalPortion / 2 + profitShare p p.investment2 + p.profitDifference) :
  p.equalPortion = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_division_l517_51740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_cost_price_theorem_l517_51738

/-- Calculates the original cost price per metre of cloth before tax and discount --/
noncomputable def original_cost_price (total_price : ℝ) (total_metres : ℝ) (loss_per_metre : ℝ) 
  (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let selling_price_per_metre := total_price / total_metres
  let cost_price_after_loss := selling_price_per_metre + loss_per_metre
  let cost_price_before_discount := cost_price_after_loss / (1 - discount_rate)
  cost_price_before_discount / (1 + tax_rate)

/-- The original cost price for one metre of cloth before tax and discount is approximately $65.31 --/
theorem original_cost_price_theorem :
  let total_price : ℝ := 45000
  let total_metres : ℝ := 750
  let loss_per_metre : ℝ := 7
  let discount_rate : ℝ := 0.05
  let tax_rate : ℝ := 0.08
  abs (original_cost_price total_price total_metres loss_per_metre discount_rate tax_rate - 65.31) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_cost_price_theorem_l517_51738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_overall_profit_l517_51765

/-- Represents an item with its cost price and profit/loss percentage -/
structure Item where
  cost_price : ℕ
  profit_percentage : ℤ

/-- Calculates the selling price of an item -/
def selling_price (item : Item) : ℤ :=
  item.cost_price + item.cost_price * item.profit_percentage / 100

/-- Calculates the total profit or loss for a list of items -/
def total_profit_or_loss (items : List Item) : ℤ :=
  (items.map selling_price).sum - (items.map (·.cost_price)).sum

/-- The list of items John purchased -/
def johns_items : List Item := [
  { cost_price := 15000, profit_percentage := -4 },
  { cost_price := 8000, profit_percentage := 10 },
  { cost_price := 24000, profit_percentage := 8 },
  { cost_price := 12000, profit_percentage := -6 }
]

theorem johns_overall_profit :
  total_profit_or_loss johns_items = 1400 := by
  sorry

#eval total_profit_or_loss johns_items

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_overall_profit_l517_51765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_decomposition_l517_51779

/-- The set A in the complex plane -/
def A (a c : ℝ) : Set ℂ :=
  {z | Complex.abs (z - c) + Complex.abs (z + c) = 2 * a ∧ a > c ∧ c > 0}

/-- The subset A₁ -/
def A₁ : Set ℂ :=
  {z | z = 2 + Complex.I ∨ z = 2 - Complex.I ∨ z = -2 + Complex.I ∨ z = -2 - Complex.I}

/-- The subset A₂ -/
def A₂ : Set ℂ :=
  {z | z.re^2 + z.im^2 > 5 ∧ Complex.abs z.im < 1}

/-- The subset A₃ -/
def A₃ : Set ℂ :=
  {z | z.re^2 + z.im^2 < 5 ∧ Complex.abs z.im > 1}

/-- The main theorem -/
theorem A_decomposition (a c : ℝ) (h1 : a > c) (h2 : c > 0) :
  (∀ z : ℂ, z + Complex.I ∈ A a c) → A a c = A₁ ∪ A₂ ∪ A₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_decomposition_l517_51779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l517_51746

-- Define the speed of the train in km/hr
noncomputable def train_speed_kmh : ℝ := 54

-- Define the time taken to cross the electric pole in seconds
noncomputable def crossing_time_s : ℝ := 5

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Theorem to prove the length of the train
theorem train_length_proof :
  let train_speed_ms := train_speed_kmh * kmh_to_ms
  train_speed_ms * crossing_time_s = 75 := by
  -- Unfold the definitions
  unfold train_speed_kmh crossing_time_s kmh_to_ms
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l517_51746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_extrema_l517_51782

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin ((Real.pi / 2) * x + Real.pi / 5)

theorem min_distance_extrema (x₁ x₂ : ℝ) 
  (h : ∀ x : ℝ, f x₁ ≤ f x ∧ f x ≤ f x₂) :
  ∃ (y₁ y₂ : ℝ), (∀ x : ℝ, f y₁ ≤ f x ∧ f x ≤ f y₂) ∧ |y₁ - y₂| = 2 ∧
  ∀ z₁ z₂ : ℝ, (∀ x : ℝ, f z₁ ≤ f x ∧ f x ≤ f z₂) → |z₁ - z₂| ≥ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_extrema_l517_51782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_as_fraction_l517_51737

/-- The repeating decimal 0.3666... as a real number -/
noncomputable def repeating_decimal : ℝ := 0.3 + (2 / 3) * (1 / 10)

/-- Theorem stating that the repeating decimal 0.3666... is equal to 33/90 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 33 / 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_as_fraction_l517_51737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l517_51719

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
noncomputable def train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : ℝ :=
  speed * (1000 / 3600) * time - bridge_length

/-- Proves that a train traveling at 72 km/hr that takes 14.248860091192705 seconds to cross a 175 m long bridge has a length of approximately 110.9772018238541 m. -/
theorem train_length_calculation :
  let speed := (72 : ℝ)
  let time := (14.248860091192705 : ℝ)
  let bridge_length := (175 : ℝ)
  abs (train_length speed time bridge_length - 110.9772018238541) < 0.0000000000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l517_51719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_product_formula_l517_51714

/-- For a geometric progression with n terms, first term b, and common ratio k -/
structure GeometricProgression where
  n : ℕ
  b : ℝ
  k : ℝ

/-- The product of terms in a geometric progression -/
noncomputable def product (gp : GeometricProgression) : ℝ :=
  gp.b^gp.n * gp.k^(gp.n * (gp.n - 1) / 2)

/-- The sum of terms in a geometric progression -/
noncomputable def sum (gp : GeometricProgression) : ℝ :=
  gp.b * (1 - gp.k^gp.n) / (1 - gp.k)

/-- The sum of reciprocals of terms in a geometric progression -/
noncomputable def sumReciprocals (gp : GeometricProgression) : ℝ :=
  (gp.k^gp.n - 1) / (gp.b * (gp.k - 1))

theorem geometric_progression_product_formula (gp : GeometricProgression) 
    (h : gp.n = 5) :
    product gp = (sum gp * sumReciprocals gp)^((gp.n - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_product_formula_l517_51714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_mutually_known_exist_four_mutually_known_exist_nine_l517_51724

/-- Represents a group of people and their acquaintances. -/
structure PeopleGroup where
  people : Finset Nat
  knows : Nat → Nat → Prop

/-- The property that among any three people, at least two know each other. -/
def AtLeastTwoKnowEachOther (g : PeopleGroup) : Prop :=
  ∀ a b c, a ∈ g.people → b ∈ g.people → c ∈ g.people →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    g.knows a b ∨ g.knows b c ∨ g.knows a c

/-- The existence of a subset of four people where any two know each other. -/
def ExistsFourMutuallyKnown (g : PeopleGroup) : Prop :=
  ∃ a b c d, a ∈ g.people ∧ b ∈ g.people ∧ c ∈ g.people ∧ d ∈ g.people ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    g.knows a b ∧ g.knows a c ∧ g.knows a d ∧
    g.knows b c ∧ g.knows b d ∧ g.knows c d

/-- The main theorem to be proved. -/
theorem four_mutually_known_exist
  (g : PeopleGroup)
  (h1 : g.people.card = 10)
  (h2 : AtLeastTwoKnowEachOther g) :
  ExistsFourMutuallyKnown g :=
sorry

/-- The same theorem with 9 people instead of 10. -/
theorem four_mutually_known_exist_nine
  (g : PeopleGroup)
  (h1 : g.people.card = 9)
  (h2 : AtLeastTwoKnowEachOther g) :
  ExistsFourMutuallyKnown g :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_mutually_known_exist_four_mutually_known_exist_nine_l517_51724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_l517_51735

def sequence_a : ℕ → ℝ
  | 0 => 98  -- Adding the case for 0 to cover all natural numbers
  | 1 => 98  -- Derived from the condition a_2 - a_1 = 4
  | 2 => 102
  | (n + 3) => sequence_a (n + 2) + 4 * (n + 2)

theorem min_value_of_sequence :
  ∃ k : ℕ, k > 0 ∧ sequence_a k / k = 26 ∧ ∀ m : ℕ, m > 0 → sequence_a m / m ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_l517_51735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_goals_approx_l517_51708

open Real

/-- The parameter for the Poisson distribution representing the expected number of goals -/
noncomputable def lambda : ℝ := 2.8

/-- The probability that an even number of goals will be scored -/
noncomputable def prob_even_goals : ℝ := (1 + Real.exp (-2 * lambda)) / 2

/-- Theorem stating that the probability of an even number of goals is approximately 0.502 -/
theorem prob_even_goals_approx :
  |prob_even_goals - 0.502| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_goals_approx_l517_51708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_expression_l517_51763

theorem fourth_quadrant_trig_expression (α : ℝ) : 
  α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) →  -- fourth quadrant
  Real.cos α = 3 / 5 → 
  (1 + Real.sqrt 2 * Real.cos (2 * α - Real.pi / 4)) / Real.sin (α + Real.pi / 2) = -2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_quadrant_trig_expression_l517_51763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a7_range_l517_51741

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

theorem a7_range (seq : ArithmeticSequence) 
  (h1 : S seq 4 ≥ 10)
  (h2 : S seq 5 ≤ 15)
  (h3 : S seq 7 ≥ 21) :
  3 ≤ seq.a 7 ∧ seq.a 7 ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a7_range_l517_51741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_common_divisors_36_18_l517_51712

theorem sum_common_divisors_36_18 : 
  (Finset.filter (fun x => x ∣ 36 ∧ x ∣ 18) (Finset.range 37)).sum id = 39 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_common_divisors_36_18_l517_51712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_ball_labels_l517_51777

/-- We can verify that 13 balls are correctly labeled in 3 weighings. -/
theorem verify_ball_labels : 
  ∃ (weighing_strategy : ℕ → ℕ → Prop),
  (∀ labeling : Fin 13 → Fin 13, 
    (∀ i j : Fin 13, i < j → labeling i < labeling j) →
    ∃ (w1 w2 w3 : Bool), 
      weighing_strategy (w1.toNat + 2 * w2.toNat + 4 * w3.toNat) 0 ∧
      (∀ i : Fin 13, labeling i = i)) :=
by
  -- The proof goes here
  sorry

#check verify_ball_labels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_ball_labels_l517_51777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_in_unit_square_l517_51723

open Set
open Real

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a square
def UnitSquare : Set Point :=
  {p | 0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1}

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem five_points_in_unit_square :
  ∀ (points : Finset Point),
    points.card = 5 →
    (∀ p, p ∈ points → p ∈ UnitSquare) →
    ∃ p q, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ distance p q < 1 / Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_in_unit_square_l517_51723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l517_51739

theorem square_perimeter_ratio (s : ℝ) (h : s > 0) : 
  (4 * (11 * s * Real.sqrt 2 / Real.sqrt 2)) / (4 * s) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_ratio_l517_51739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_is_16_l517_51726

/-- A right triangular prism with the sum of areas of three mutually adjacent faces equal to 24 -/
structure RightTriangularPrism where
  -- Base sides
  a : ℝ
  b : ℝ
  -- Height of the prism
  h : ℝ
  -- Angle between sides a and b
  θ : ℝ
  -- All dimensions are positive
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h
  -- θ is between 0 and π
  θ_range : 0 < θ ∧ θ < π
  -- Sum of areas of three mutually adjacent faces is 24
  area_sum : a * h + b * h + 1/2 * a * b * Real.sin θ = 24

/-- The volume of a right triangular prism -/
noncomputable def volume (p : RightTriangularPrism) : ℝ :=
  1/2 * p.a * p.b * p.h * Real.sin p.θ

/-- Theorem: The maximum volume of a right triangular prism with the given conditions is 16 -/
theorem max_volume_is_16 (p : RightTriangularPrism) : volume p ≤ 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_is_16_l517_51726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_OP_OQ_l517_51734

-- Define the curves in polar coordinates
noncomputable def C₁ (θ : Real) : Real := 4 * Real.cos θ
noncomputable def C₂ (θ : Real) : Real := 4 * Real.sin θ

-- Define the rays
noncomputable def l₁ (α : Real) : Real := α
noncomputable def l₂ (α : Real) : Real := α - Real.pi/6

-- Define the intersection points
noncomputable def P (α : Real) : Real := C₁ (l₁ α)
noncomputable def Q (α : Real) : Real := C₂ (l₂ α)

-- State the theorem
theorem max_product_OP_OQ :
  ∃ (max : Real), ∀ (α : Real), 0 < α → α < Real.pi/2 →
    P α * Q α ≤ max ∧ ∃ (α₀ : Real), 0 < α₀ → α₀ < Real.pi/2 → P α₀ * Q α₀ = max ∧ max = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_OP_OQ_l517_51734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l517_51755

theorem equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x - (2 : ℝ)^x - 6 = 0 ∧ x = Real.log 3 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l517_51755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_values_order_l517_51736

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + (abs b) * x + c

theorem parabola_y_values_order 
  (a b c : ℝ) 
  (h_symmetry_axis : -((abs b) / (2 * a)) = -1) 
  (y₁ y₂ y₃ : ℝ) 
  (h_point1 : quadratic_function a b c (-14/3) = y₁)
  (h_point2 : quadratic_function a b c (5/2) = y₂)
  (h_point3 : quadratic_function a b c 3 = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_y_values_order_l517_51736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l517_51764

-- Define the parallelogram ABCD
noncomputable def parallelogram_ABCD : Set (ℝ × ℝ) := sorry

-- Define the area of parallelogram ABCD
noncomputable def area_ABCD : ℝ := sorry

-- Define a function to calculate the area of triangle MCD
noncomputable def area_MCD (M : ℝ × ℝ) : ℝ := sorry

-- Define the set of points M that satisfy the condition
noncomputable def satisfying_points : Set (ℝ × ℝ) :=
  {M ∈ parallelogram_ABCD | area_MCD M < area_ABCD / 3}

-- State the theorem
theorem probability_theorem :
  (MeasureTheory.volume satisfying_points) / (MeasureTheory.volume parallelogram_ABCD) = 2/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l517_51764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discounts_is_seven_l517_51788

/-- Represents the maximum number of discounts that can be offered for a commodity. -/
def max_discounts (cost_price marked_price : ℚ) (min_profit_margin : ℚ) : ℕ :=
  (((marked_price - (1 + min_profit_margin) * cost_price) / (marked_price / 10)).floor).toNat

/-- Theorem stating that the maximum number of discounts is 7 for the given conditions. -/
theorem max_discounts_is_seven :
  max_discounts 700 1100 (1/10) = 7 := by
  sorry

#eval max_discounts 700 1100 (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discounts_is_seven_l517_51788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gihuns_farm_cows_l517_51757

/-- The number of cows on Gihun's farm -/
def num_cows : ℕ := 500

/-- The number of pigs on Gihun's farm -/
def num_pigs : ℕ := 100

/-- The relationship between the number of cows and pigs -/
axiom cow_pig_relation : (3/4 * (num_cows : ℝ) - 25) / 7 + 50 = (num_pigs : ℝ)

/-- Theorem stating the number of cows on Gihun's farm -/
theorem gihuns_farm_cows : num_cows = 500 := by
  -- The proof is omitted for now
  sorry

#eval num_cows -- This will print 500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gihuns_farm_cows_l517_51757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_at_t_4_l517_51775

def line (t : ℝ) : ℝ × ℝ × ℝ :=
  sorry -- We'll leave the definition as sorry for now

theorem line_at_t_4 (h1 : line (-1) = (2, 5, 11))
                    (h2 : line 1 = (1, 2, 4))
                    (h3 : line 0 = (3, 7, 15)) :
  line 4 = (-5, -13, -29) := by
  sorry -- We'll leave the proof as sorry for now

#check line_at_t_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_at_t_4_l517_51775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l517_51733

/-- The equation of a parabola -/
noncomputable def parabola_equation (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

/-- The focus of a parabola -/
noncomputable def focus : ℝ × ℝ := (1, 23/8)

/-- Theorem: The focus of the parabola y = -2x^2 + 4x + 1 is (1, 23/8) -/
theorem parabola_focus :
  focus = (1, 23/8) ∧
  ∀ x y : ℝ, y = parabola_equation x ↔ 
    (y - focus.2) = (-2) * ((x - focus.1)^2 + 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l517_51733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_mappings_count_specific_mappings_l517_51784

/-- Given two finite sets P and Q, this theorem states that the number of possible
    mappings from P to Q is equal to the cardinality of Q raised to the power of
    the cardinality of P. -/
theorem count_mappings (P Q : Finset ℕ) (hP : P = {0, 1}) (hQ : Q = {0, 1, 2}) :
  Fintype.card (P → Q) = 9 := by
  sorry

/-- This corollary applies the general theorem to the specific case
    where P = {a, b} and Q = {-1, 0, 1}. -/
theorem count_specific_mappings :
  let P : Finset ℕ := {0, 1}
  let Q : Finset ℤ := {-1, 0, 1}
  Fintype.card (P → Q) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_mappings_count_specific_mappings_l517_51784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_block_height_l517_51717

/-- Represents the properties of a cylindrical container and a conical block --/
structure Container :=
  (container_diameter : ℝ)
  (block_diameter : ℝ)
  (water_level_drop : ℝ)

/-- Calculates the height of a conical block based on water displacement --/
noncomputable def block_height (c : Container) : ℝ :=
  (c.water_level_drop * c.container_diameter^2) / c.block_diameter^2

/-- Theorem stating that the height of the conical block is 5 cm --/
theorem conical_block_height :
  let c := Container.mk 10 8 3.2
  block_height c = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conical_block_height_l517_51717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_C₁_no_intersection_C_contained_in_C₁_l517_51750

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sqrt 2 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 2 * Real.cos θ * Real.sin θ)

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the locus C₁
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ :=
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Theorem stating that C and C₁ do not intersect
theorem C_C₁_no_intersection :
  ∀ θ₁ θ₂ : ℝ, C θ₁ ≠ C₁ θ₂ :=
by
  sorry

-- Theorem stating that C is contained within C₁
theorem C_contained_in_C₁ :
  ∀ θ : ℝ, ∃ θ' : ℝ, norm (C θ - A) < norm (C₁ θ' - A) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_C₁_no_intersection_C_contained_in_C₁_l517_51750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l517_51706

/-- A geometric sequence with terms a_n -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (a : ℕ → ℝ) : ℝ :=
  (a 6) / (a 5)

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  geometric_sequence a → a 5 = 1 → a 8 = 8 → common_ratio a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l517_51706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_no_maximum_l517_51716

-- Define the function
noncomputable def f (x : ℝ) := (1/2) * x^2 - 9 * Real.log x

-- State the theorem
theorem f_minimum_no_maximum :
  (∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x₀) ∧
  (∀ (x₀ : ℝ), ¬(∀ (x : ℝ), x > 0 → f x ≤ f x₀)) ∧
  (∃ (x₀ : ℝ), x₀ = 3 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_no_maximum_l517_51716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_contains_geometric_iff_rational_ratio_l517_51756

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a d : ℚ) : ℕ → ℚ :=
  fun n => a + n • d

/-- A geometric progression is a sequence where each term after the first is found by
    multiplying the previous term by a fixed, non-zero number. -/
def GeometricProgression (a r : ℚ) : ℕ → ℚ :=
  fun n => a * r^n

/-- A subsequence of a sequence is a sequence that can be derived from the original
    sequence by deleting some or no elements without changing the order of the
    remaining elements. -/
def IsSubsequence {α : Type*} (f g : ℕ → α) : Prop :=
  ∃ h : ℕ → ℕ, StrictMono h ∧ f = g ∘ h

/-- Main theorem: An arithmetic progression contains a geometric progression as a
    subsequence if and only if the ratio of its first term to its common difference
    is rational. -/
theorem arithmetic_contains_geometric_iff_rational_ratio (a d : ℚ) (hd : d ≠ 0) :
  (∃ (b r : ℚ) (hr : r ≠ 1), IsSubsequence (GeometricProgression b r) (ArithmeticProgression a d)) ↔
  ∃ (p q : ℤ), q ≠ 0 ∧ a / d = p / q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_contains_geometric_iff_rational_ratio_l517_51756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l517_51768

/-- Given a parabola and a line intersecting it, prove the equation of the directrix -/
theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ (x y : ℝ), y = x^2 / (2*p)) →
  (∃ t : ℝ, A.2 = A.1 + t ∧ B.2 = B.1 + t) →
  A.1^2 = 2*p*A.2 →
  B.1^2 = 2*p*B.2 →
  (A.1 + B.1) / 2 = 2 →
  ∃ y : ℝ, ∀ x : ℝ, y = -p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l517_51768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l517_51773

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 1/2

theorem min_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

#check min_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l517_51773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_unique_coefficients_l517_51793

/-- A polynomial function of degree 3 -/
def cubic_polynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

/-- The condition that |f(x)| ≤ 1 for |x| ≤ 1 -/
def bounded_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |x| ≤ 1 → |f x| ≤ 1

theorem cubic_polynomial_unique_coefficients :
  ∀ a b c d : ℝ,
  let f := cubic_polynomial a b c d
  bounded_on_unit_interval f ∧ f 2 = 26 →
  a = 4 ∧ b = 0 ∧ c = -3 ∧ d = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_unique_coefficients_l517_51793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l517_51754

-- Define is_triangle as a predicate
def is_triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧ 
  A + B + C = Real.pi

theorem triangle_classification (a b c A B C : ℝ) 
  (h_triangle : is_triangle a b c A B C)
  (h_cos : a * Real.cos A = b * Real.cos B) :
  (A = B) ∨ (A + B = Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l517_51754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coeff_equation_l517_51781

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Use the existing logarithm function from Mathlib
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the function for the maximum binomial coefficient in (2x-x^(lg x))^8
noncomputable def max_coeff (x : ℝ) : ℝ := 
  (binomial_coeff 8 4 : ℝ) * (2*x)^4 * (x^(log2 x))^4

-- State the theorem
theorem max_coeff_equation (x : ℝ) (h : x > 0): 
  (max_coeff x = 1120) → (x = 1/10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coeff_equation_l517_51781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_average_speed_not_necessarily_constant_l517_51778

/-- A pedestrian's walk with a given duration and distance covered per hour interval -/
structure PedestrianWalk where
  duration : ℝ
  distance_per_hour : ℝ

/-- The average speed of a pedestrian's walk -/
noncomputable def average_speed (walk : PedestrianWalk) : ℝ :=
  (walk.distance_per_hour * walk.duration) / walk.duration

/-- Theorem stating that a pedestrian walking for 3.5 hours and covering 5 km every hour
    does not necessarily have an average speed of 5 km/h -/
theorem pedestrian_average_speed_not_necessarily_constant
  (walk : PedestrianWalk)
  (h1 : walk.duration = 3.5)
  (h2 : walk.distance_per_hour = 5) :
  ¬ (∀ w : PedestrianWalk, w.duration = 3.5 ∧ w.distance_per_hour = 5 → average_speed w = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_average_speed_not_necessarily_constant_l517_51778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_irrational_in_S_l517_51758

def S : Set ℝ := {-3.14, 0, -Real.pi, 22/3, 1.12112}

theorem one_irrational_in_S : ∃! x, x ∈ S ∧ Irrational x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_irrational_in_S_l517_51758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l517_51713

/-- Define what it means for an equation to represent an ellipse -/
def IsEllipse (x y : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

/-- The necessary but not sufficient condition for the equation x²/(4+m) + y²/(2-m) = 1 to represent an ellipse -/
theorem ellipse_condition (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (4 + m) + y^2 / (2 - m) = 1 → IsEllipse x y) → 
  m ∈ Set.Ioo (-4 : ℝ) 2 :=
by
  sorry

#check ellipse_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_condition_l517_51713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polyhedron_with_short_diagonals_l517_51742

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  -- We don't need to specify the internal structure for this problem
  mk :: -- Constructor

/-- A diagonal of a polyhedron is a line segment that connects two non-adjacent vertices. -/
def diagonal (p : ConvexPolyhedron) : Type := sorry

/-- An edge of a polyhedron is a line segment where two faces meet. -/
def edge (p : ConvexPolyhedron) : Type := sorry

/-- The length of a line segment in a polyhedron. -/
noncomputable def length {p : ConvexPolyhedron} (s : diagonal p ⊕ edge p) : ℝ := sorry

/-- Theorem: There exists a convex polyhedron where every diagonal is shorter than any edge. -/
theorem exists_polyhedron_with_short_diagonals :
  ∃ (p : ConvexPolyhedron), 
    (∀ (d : diagonal p) (e : edge p), length (Sum.inl d) < length (Sum.inr e)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polyhedron_with_short_diagonals_l517_51742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_the_a_l517_51701

/-- Represents the article choices --/
inductive Article
  | the
  | an
  | a

/-- Represents the correct answer for the problem --/
def correctAnswer : Article × Article := (Article.the, Article.a)

/-- Checks if a given answer matches the correct answer --/
def isCorrectAnswer (answer : Article × Article) : Prop :=
  answer = correctAnswer

/-- Theorem stating that the correct answer is (the, a) --/
theorem correct_answer_is_the_a :
  isCorrectAnswer (Article.the, Article.a) := by
  -- The proof would go here, but we'll use sorry for now
  sorry

#check correct_answer_is_the_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_is_the_a_l517_51701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l517_51796

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  investment_ratio_a : ℚ
  investment_ratio_b : ℚ
  investment_ratio_c : ℚ
  return_ratio_a : ℚ
  return_ratio_b : ℚ
  return_ratio_c : ℚ

/-- Calculates the total earnings given investment data and the earnings difference between b and a -/
def calculate_total_earnings (data : InvestmentData) (earnings_diff_b_a : ℚ) : ℚ :=
  let xy := earnings_diff_b_a * 50
  (58 * xy) / 100

/-- Theorem stating that given the specific investment ratios, return ratios, and earnings difference,
    the total earnings will be 10150 -/
theorem total_earnings_proof (data : InvestmentData) (h1 : data.investment_ratio_a = 3)
    (h2 : data.investment_ratio_b = 4) (h3 : data.investment_ratio_c = 5)
    (h4 : data.return_ratio_a = 6) (h5 : data.return_ratio_b = 5) (h6 : data.return_ratio_c = 4)
    (h7 : earnings_diff_b_a = 350) :
    calculate_total_earnings data earnings_diff_b_a = 10150 := by
  sorry

#eval calculate_total_earnings
  { investment_ratio_a := 3, investment_ratio_b := 4, investment_ratio_c := 5,
    return_ratio_a := 6, return_ratio_b := 5, return_ratio_c := 4 }
  350

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l517_51796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_fiftieth_digit_is_five_l517_51720

def digit_sequence (n : ℕ) : List ℕ :=
  (List.range (n - 1 + 1)).reverse.bind (λ i => i.repr.toList.map (λ c => c.toNat - 48))

theorem two_hundred_fiftieth_digit_is_five :
  (digit_sequence 150).get? 249 = some 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hundred_fiftieth_digit_is_five_l517_51720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_fraction_l517_51749

theorem rectangle_circle_fraction :
  ∀ (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ),
    square_area = 3600 →
    rectangle_area = 240 →
    rectangle_breadth = 10 →
    (rectangle_area / rectangle_breadth) / (Real.sqrt square_area) = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_fraction_l517_51749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_theorem_l517_51707

/-- A balanced chessboard is a 100x100 matrix of non-negative real numbers where each row and column sums to 1 -/
def BalancedChessboard : Type := Matrix (Fin 100) (Fin 100) ℝ

/-- A cell selection is a function from Fin 100 to Fin 100, representing the selected column for each row -/
def CellSelection : Type := Fin 100 → Fin 100

theorem chessboard_theorem (board : BalancedChessboard) : 
  ∃ (selection : CellSelection), 
    (∀ i j : Fin 100, i ≠ j → selection i ≠ selection j) ∧ 
    (∀ i : Fin 100, 0 ≤ board i (selection i) ∧ board i (selection i) ≥ (1 : ℝ) / 2601) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_theorem_l517_51707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l517_51762

def M : Set ℝ := {x : ℝ | Real.log (x^2) = Real.log x ∧ x > 0}

def N : Set ℝ := {n : ℝ | n ∈ Set.Icc (-1 : ℝ) 3 ∧ ∃ (m : ℤ), n = m}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l517_51762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l517_51751

theorem problem_1 : 
  -|(-(5 : ℤ))| + (3.14 - Real.pi)^(0 : ℕ) + 8 * 2^(-(2 : ℤ)) - (-1 : ℤ)^2022 = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l517_51751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_x_minus_y_plus_3_eq_0_l517_51759

noncomputable def slope_angle (a b : ℝ) : ℝ :=
  Real.arctan (a / -b)

structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

theorem slope_angle_of_x_minus_y_plus_3_eq_0 :
  let line : LineEquation := { a := 1, b := -1, c := 3 }
  slope_angle line.a line.b = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_x_minus_y_plus_3_eq_0_l517_51759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_basic_terms_div_four_l517_51727

/-- Represents a table filled with +1 or -1 -/
def Table (n : ℕ) := Fin n → Fin n → Int

/-- A basic term is a product of n cells, one from each row and column -/
def BasicTerm (n : ℕ) (t : Table n) (p : Equiv.Perm (Fin n)) : Int :=
  (Finset.univ.prod fun i => t i (p i))

/-- The sum of all basic terms in the table -/
def SumBasicTerms (n : ℕ) (t : Table n) : Int :=
  (Finset.univ.sum fun p : Equiv.Perm (Fin n) => BasicTerm n t p)

/-- The main theorem: for n ≥ 4, the sum of all basic terms is divisible by 4 -/
theorem sum_basic_terms_div_four {n : ℕ} (h : n ≥ 4) (t : Table n) :
  4 ∣ SumBasicTerms n t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_basic_terms_div_four_l517_51727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_sixth_power_eq_one_l517_51753

noncomputable def w : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

theorem w_sixth_power_eq_one : w^6 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_sixth_power_eq_one_l517_51753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_defined_log3_fold_l517_51725

-- Define the tower function of threes
def T : ℕ → ℕ
  | 0 => 3
  | 1 => 3
  | (n + 1) => 3^(T n)

-- Define A and B
def A : ℕ := T 4
def B : ℕ := A^A

-- Define the k-fold composition of log_3
noncomputable def log3_fold : ℕ → ℝ → ℝ
  | 0, x => x
  | (k + 1), x => log3_fold k (Real.log x / Real.log 3)

-- Theorem statement
theorem largest_defined_log3_fold :
  (∀ k : ℕ, k ≤ 4 → log3_fold k (B : ℝ) > 0) ∧
  log3_fold 5 (B : ℝ) ≤ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_defined_log3_fold_l517_51725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l517_51767

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

-- State the theorem
theorem f_range : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
  ∃ y ∈ Set.Icc (-3/2) 3, f x = y ∧ 
  ∀ z, f x = z → z ∈ Set.Icc (-3/2) 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l517_51767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_milk_production_l517_51745

noncomputable section

variable (y : ℝ)

/-- The milk production function -/
def milk_production (cows days : ℝ) : ℝ := (y + 2) * days / (y * (y + 3))

/-- Theorem about cow milk production -/
theorem cow_milk_production :
  let new_cows := y + 4
  let new_milk := y + 6
  (new_milk / (milk_production y new_cows 1)) = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by
  -- Unfold definitions
  simp [milk_production]
  -- The rest of the proof is omitted
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cow_milk_production_l517_51745
