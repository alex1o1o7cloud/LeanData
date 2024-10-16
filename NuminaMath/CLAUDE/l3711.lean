import Mathlib

namespace NUMINAMATH_CALUDE_sin_arccos_circle_l3711_371141

theorem sin_arccos_circle (x y : ℝ) :
  y = Real.sin (Real.arccos x) ↔ x^2 + y^2 = 1 ∧ x ∈ Set.Icc (-1) 1 ∧ y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_circle_l3711_371141


namespace NUMINAMATH_CALUDE_equation_with_integer_roots_l3711_371124

/-- Given that the equation (x-a)(x-8) - 1 = 0 has two integer roots, prove that a = 8 -/
theorem equation_with_integer_roots (a : ℤ) :
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ - a) * (x₁ - 8) - 1 = 0 ∧ (x₂ - a) * (x₂ - 8) - 1 = 0) →
  a = 8 := by
  sorry


end NUMINAMATH_CALUDE_equation_with_integer_roots_l3711_371124


namespace NUMINAMATH_CALUDE_green_peaches_count_l3711_371136

/-- Represents a basket of peaches -/
structure Basket where
  total : Nat
  red : Nat
  green : Nat

/-- The number of green peaches in a basket -/
def greenPeaches (b : Basket) : Nat := b.green

/-- Theorem: Given a basket with 10 total peaches and 7 red peaches, 
    the number of green peaches is 3 -/
theorem green_peaches_count (b : Basket) 
  (h1 : b.total = 10) 
  (h2 : b.red = 7) 
  (h3 : b.green = b.total - b.red) : 
  greenPeaches b = 3 := by
  sorry

#check green_peaches_count

end NUMINAMATH_CALUDE_green_peaches_count_l3711_371136


namespace NUMINAMATH_CALUDE_average_cost_before_gratuity_l3711_371111

theorem average_cost_before_gratuity
  (num_individuals : ℕ)
  (total_bill_with_gratuity : ℚ)
  (gratuity_rate : ℚ)
  (h1 : num_individuals = 9)
  (h2 : total_bill_with_gratuity = 756)
  (h3 : gratuity_rate = 1/5) :
  (total_bill_with_gratuity / (1 + gratuity_rate)) / num_individuals = 70 := by
sorry

end NUMINAMATH_CALUDE_average_cost_before_gratuity_l3711_371111


namespace NUMINAMATH_CALUDE_line_equation_l3711_371105

/-- The ellipse E defined by the equation x^2/4 + y^2/2 = 1 -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 2 = 1}

/-- A line intersecting the ellipse E -/
def l : Set (ℝ × ℝ) := sorry

/-- Point A on the ellipse E and line l -/
def A : ℝ × ℝ := sorry

/-- Point B on the ellipse E and line l -/
def B : ℝ × ℝ := sorry

/-- The midpoint of AB is (1/2, -1) -/
axiom midpoint_AB : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = -1

/-- Theorem stating that the equation of line l is x - 4y - 9/2 = 0 -/
theorem line_equation : l = {p : ℝ × ℝ | p.1 - 4 * p.2 - 9/2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3711_371105


namespace NUMINAMATH_CALUDE_height_C_is_9_l3711_371165

/-- A regular octagon with pillars at each vertex -/
structure OctagonWithPillars where
  /-- Side length of the octagon -/
  side_length : ℝ
  /-- Height of pillar at vertex A -/
  height_A : ℝ
  /-- Height of pillar at vertex E -/
  height_E : ℝ
  /-- Height of pillar at vertex G -/
  height_G : ℝ

/-- The height of the pillar at vertex C given the heights at A, E, and G -/
def height_C (o : OctagonWithPillars) : ℝ :=
  sorry

/-- Theorem stating that the height of pillar C is 9 meters given the specified conditions -/
theorem height_C_is_9 (o : OctagonWithPillars) 
  (h_side : o.side_length = 8)
  (h_A : o.height_A = 15)
  (h_E : o.height_E = 11)
  (h_G : o.height_G = 13) :
  height_C o = 9 :=
sorry

end NUMINAMATH_CALUDE_height_C_is_9_l3711_371165


namespace NUMINAMATH_CALUDE_technician_salary_l3711_371125

theorem technician_salary (total_workers : Nat) (technicians : Nat) (avg_salary : ℕ) 
  (non_tech_avg : ℕ) (h1 : total_workers = 12) (h2 : technicians = 6) 
  (h3 : avg_salary = 9000) (h4 : non_tech_avg = 6000) : 
  (total_workers * avg_salary - (total_workers - technicians) * non_tech_avg) / technicians = 12000 :=
sorry

end NUMINAMATH_CALUDE_technician_salary_l3711_371125


namespace NUMINAMATH_CALUDE_circle_properties_l3711_371177

/-- A circle with center on the y-axis, radius 1, passing through (1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

theorem circle_properties :
  ∃ (b : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ x^2 + (y - b)^2 = 1) ∧
    (circle_equation 1 2) ∧
    (∀ x y : ℝ, circle_equation x y → x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3711_371177


namespace NUMINAMATH_CALUDE_blender_sunday_price_l3711_371158

/-- The Sunday price of a blender after applying discounts -/
theorem blender_sunday_price (original_price : ℝ) (regular_discount : ℝ) (sunday_discount : ℝ) :
  original_price = 250 →
  regular_discount = 0.60 →
  sunday_discount = 0.25 →
  original_price * (1 - regular_discount) * (1 - sunday_discount) = 75 := by
sorry

end NUMINAMATH_CALUDE_blender_sunday_price_l3711_371158


namespace NUMINAMATH_CALUDE_committee_formation_count_l3711_371142

theorem committee_formation_count :
  let dept_A : Finset ℕ := Finset.range 6
  let dept_B : Finset ℕ := Finset.range 7
  let dept_C : Finset ℕ := Finset.range 5
  (dept_A.card * dept_B.card * dept_C.card : ℕ) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l3711_371142


namespace NUMINAMATH_CALUDE_sin_equality_proof_l3711_371108

theorem sin_equality_proof (m : ℤ) : 
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (945 * π / 180) → m = -135 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l3711_371108


namespace NUMINAMATH_CALUDE_root_equation_solution_l3711_371191

theorem root_equation_solution (a : ℚ) : 
  ((-2 : ℚ)^2 - a * (-2) + 7 = 0) → a = -11/2 := by
sorry

end NUMINAMATH_CALUDE_root_equation_solution_l3711_371191


namespace NUMINAMATH_CALUDE_xy_plus_y_squared_l3711_371185

theorem xy_plus_y_squared (x y : ℝ) (h : x * (x + y) = x^2 + 3*y + 12) :
  x*y + y^2 = y^2 + 3*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_plus_y_squared_l3711_371185


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l3711_371121

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  trace_effect_to_cause : Bool
  start_from_inequality : Bool

/-- A condition in the context of inequality proofs -/
inductive Condition
  | necessary
  | sufficient
  | necessary_and_sufficient
  | necessary_or_sufficient

/-- The condition sought by the analysis method -/
def condition_sought (method : AnalysisMethod) : Condition :=
  Condition.sufficient

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition (method : AnalysisMethod) 
  (h1 : method.trace_effect_to_cause = true) 
  (h2 : method.start_from_inequality = true) : 
  condition_sought method = Condition.sufficient := by sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l3711_371121


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l3711_371163

theorem polygon_sides_from_interior_angle (interior_angle : ℝ) (n : ℕ) :
  interior_angle = 108 →
  (n : ℝ) * (180 - interior_angle) = 360 →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l3711_371163


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3711_371196

theorem complex_fraction_simplification :
  let z : ℂ := (3 - 2*I) / (1 + 5*I)
  z = -7/26 - 17/26*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3711_371196


namespace NUMINAMATH_CALUDE_propositions_truth_l3711_371102

-- Define the necessary geometric concepts
def Line : Type := sorry
def Plane : Type := sorry

def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- Define the propositions
def proposition_1 (p1 p2 p3 : Plane) (l1 l2 : Line) : Prop :=
  line_in_plane l1 p1 → line_in_plane l2 p1 →
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2

def proposition_2 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular_line_plane l p1 → line_in_plane l p2 → perpendicular p1 p2

def proposition_3 (l1 l2 l3 : Line) : Prop :=
  perpendicular_line_plane l1 l3 → perpendicular_line_plane l2 l3 → parallel l1 l2

def proposition_4 (p1 p2 : Plane) (l : Line) : Prop :=
  perpendicular p1 p2 →
  line_in_plane l p1 →
  ¬perpendicular_line_plane l (line_of_intersection p1 p2) →
  ¬perpendicular_line_plane l p2

-- Theorem stating which propositions are true and which are false
theorem propositions_truth : 
  (∃ p1 p2 p3 : Plane, ∃ l1 l2 : Line, ¬proposition_1 p1 p2 p3 l1 l2) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_2 p1 p2 l) ∧
  (∃ l1 l2 l3 : Line, ¬proposition_3 l1 l2 l3) ∧
  (∀ p1 p2 : Plane, ∀ l : Line, proposition_4 p1 p2 l) :=
sorry

end NUMINAMATH_CALUDE_propositions_truth_l3711_371102


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3711_371131

theorem quadratic_factorization (x y : ℝ) : 5*x^2 + 6*x*y - 8*y^2 = (x + 2*y)*(5*x - 4*y) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3711_371131


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3711_371137

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 145 →
  bridge_length = 230 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3711_371137


namespace NUMINAMATH_CALUDE_complex_modulus_l3711_371110

theorem complex_modulus (z : ℂ) (a : ℝ) : 
  z = a + Complex.I ∧ z + z = 1 - 3 * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3711_371110


namespace NUMINAMATH_CALUDE_pencil_length_l3711_371171

theorem pencil_length : ∀ (L : ℝ),
  (L / 8 : ℝ) +  -- Black part
  ((7 * L / 8) / 2 : ℝ) +  -- White part
  (7 / 2 : ℝ) = L →  -- Blue part
  L = 16 := by
sorry

end NUMINAMATH_CALUDE_pencil_length_l3711_371171


namespace NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l3711_371143

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Statement 1
theorem statement_1 : ∃ (a b : Line) (α : Plane),
  parallel_line a b ∧ contained_in b α ∧ 
  ¬(parallel_line_plane a α) ∧ ¬(contained_in a α) := by sorry

-- Statement 2
theorem statement_2 : ∃ (a b : Line) (α : Plane),
  parallel_line_plane a α ∧ parallel_line_plane b α ∧ 
  ¬(parallel_line a b) := by sorry

-- Statement 3
theorem statement_3 : ¬(∀ (a : Line) (α β : Plane),
  parallel_line_plane a α → parallel_line_plane a β → 
  (α = β ∨ ∃ (l : Line), parallel_line_plane l α ∧ parallel_line_plane l β)) := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l3711_371143


namespace NUMINAMATH_CALUDE_quadruple_solution_l3711_371146

theorem quadruple_solution (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (eq1 : a * b + c * d = 8)
  (eq2 : a * b * c * d = 8 + a + b + c + d) :
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_quadruple_solution_l3711_371146


namespace NUMINAMATH_CALUDE_power_product_cube_l3711_371154

theorem power_product_cube (R : Type*) [CommRing R] (x y : R) :
  (x * y^2)^3 = x^3 * y^6 := by sorry

end NUMINAMATH_CALUDE_power_product_cube_l3711_371154


namespace NUMINAMATH_CALUDE_route_length_l3711_371195

/-- Proves that given a round trip with total time of 1 hour, average speed of 8 miles/hour,
    and return speed of 20 miles/hour along the same path, the length of the one-way route is 4 miles. -/
theorem route_length (total_time : ℝ) (avg_speed : ℝ) (return_speed : ℝ) (route_length : ℝ) : 
  total_time = 1 →
  avg_speed = 8 →
  return_speed = 20 →
  route_length * 2 = avg_speed * total_time →
  route_length / return_speed + route_length / (route_length * 2 / total_time - return_speed) = total_time →
  route_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_route_length_l3711_371195


namespace NUMINAMATH_CALUDE_binary_11101_is_29_l3711_371145

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11101_is_29 :
  binary_to_decimal [true, false, true, true, true] = 29 := by
  sorry

end NUMINAMATH_CALUDE_binary_11101_is_29_l3711_371145


namespace NUMINAMATH_CALUDE_dividend_calculation_l3711_371101

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 16)
  (h_quotient : quotient = 9)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 149 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3711_371101


namespace NUMINAMATH_CALUDE_animal_food_cost_l3711_371149

/-- The total weekly cost for both animals' food -/
def total_weekly_cost : ℕ := 30

/-- The weekly cost of rabbit food -/
def rabbit_weekly_cost : ℕ := 12

/-- The number of weeks Julia has had the rabbit -/
def rabbit_weeks : ℕ := 5

/-- The number of weeks Julia has had the parrot -/
def parrot_weeks : ℕ := 3

/-- The total amount Julia has spent on animal food -/
def total_spent : ℕ := 114

theorem animal_food_cost :
  total_weekly_cost = rabbit_weekly_cost + (total_spent - rabbit_weekly_cost * rabbit_weeks) / parrot_weeks :=
by sorry

end NUMINAMATH_CALUDE_animal_food_cost_l3711_371149


namespace NUMINAMATH_CALUDE_triangle_theorem_l3711_371198

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinA : ℝ
  sinB : ℝ
  sinC : ℝ
  cosA : ℝ
  cosB : ℝ
  cosC : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 * t.b ∧ 
  t.sinC = 3/4 ∧ 
  t.b^2 + t.b * t.c = 2 * t.a^2

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.sinB = 3/8 ∧ t.cosB = (3 * Real.sqrt 6) / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3711_371198


namespace NUMINAMATH_CALUDE_hall_length_l3711_371170

theorem hall_length (hall_breadth : ℝ) (stone_length stone_width : ℝ) (num_stones : ℕ) :
  hall_breadth = 15 →
  stone_length = 0.3 →
  stone_width = 0.5 →
  num_stones = 3600 →
  (hall_breadth * (num_stones * stone_length * stone_width / hall_breadth)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_l3711_371170


namespace NUMINAMATH_CALUDE_job_completion_time_l3711_371180

theorem job_completion_time (x : ℝ) : 
  x > 0 →  -- A's completion time is positive
  8 * (1 / x + 1 / 20) = 1 - 0.06666666666666665 →  -- Condition after 8 days of working together
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3711_371180


namespace NUMINAMATH_CALUDE_sum_twenty_from_negative_nine_l3711_371189

/-- The sum of n consecutive integers starting from a given first term -/
def sumConsecutiveIntegers (n : ℕ) (first : ℤ) : ℤ :=
  n * (2 * first + n - 1) / 2

/-- Theorem: The sum of 20 consecutive integers starting from -9 is 10 -/
theorem sum_twenty_from_negative_nine :
  sumConsecutiveIntegers 20 (-9) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_twenty_from_negative_nine_l3711_371189


namespace NUMINAMATH_CALUDE_game_lives_per_player_l3711_371181

theorem game_lives_per_player (initial_players : ℕ) (new_players : ℕ) (total_lives : ℕ) :
  initial_players = 7 →
  new_players = 2 →
  total_lives = 63 →
  (total_lives / (initial_players + new_players) : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_game_lives_per_player_l3711_371181


namespace NUMINAMATH_CALUDE_max_cross_sum_l3711_371135

def CrossNumbers : Finset ℕ := {2, 5, 8, 11, 14}

theorem max_cross_sum :
  ∃ (a b c d e : ℕ),
    a ∈ CrossNumbers ∧ b ∈ CrossNumbers ∧ c ∈ CrossNumbers ∧ d ∈ CrossNumbers ∧ e ∈ CrossNumbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a + b + e = b + d + e ∧
    a + c + e = a + b + e ∧
    a + b + e = 36 ∧
    ∀ (x y z : ℕ),
      x ∈ CrossNumbers → y ∈ CrossNumbers → z ∈ CrossNumbers →
      x + y + z ≤ 36 :=
by sorry

end NUMINAMATH_CALUDE_max_cross_sum_l3711_371135


namespace NUMINAMATH_CALUDE_problem_solution_l3711_371183

theorem problem_solution (a b : ℕ) (ha : a = 3) (hb : b = 2) : 
  (a^(b+1))^a + (b^(a+1))^b = 19939 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3711_371183


namespace NUMINAMATH_CALUDE_candy_distribution_l3711_371117

/-- The number of pieces of candy in each of Wendy's boxes -/
def candy_per_box : ℕ := sorry

/-- The number of pieces of candy Wendy's brother has -/
def brother_candy : ℕ := 6

/-- The number of boxes Wendy has -/
def wendy_boxes : ℕ := 2

/-- The total number of pieces of candy -/
def total_candy : ℕ := 12

theorem candy_distribution :
  candy_per_box * wendy_boxes + brother_candy = total_candy ∧ candy_per_box = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3711_371117


namespace NUMINAMATH_CALUDE_total_cost_of_two_items_l3711_371167

/-- The total cost of two items is the sum of their individual costs -/
theorem total_cost_of_two_items (yoyo_cost whistle_cost : ℕ) :
  yoyo_cost = 24 → whistle_cost = 14 →
  yoyo_cost + whistle_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_two_items_l3711_371167


namespace NUMINAMATH_CALUDE_smallest_common_pet_count_l3711_371161

theorem smallest_common_pet_count : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), x > 1 ∧ 2 ∣ m ∧ x ∣ m) → 
    n ≤ m) ∧ 
  (∃ (x : ℕ), x > 1 ∧ 2 ∣ n ∧ x ∣ n) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_pet_count_l3711_371161


namespace NUMINAMATH_CALUDE_incorrect_proposition_statement_l3711_371106

theorem incorrect_proposition_statement : ∃ (p q : Prop), 
  (¬(p ∧ q)) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_statement_l3711_371106


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3711_371139

theorem shaded_area_calculation (area_ABCD area_overlap : ℝ) 
  (h1 : area_ABCD = 196)
  (h2 : area_overlap = 1)
  (h3 : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2 = 4*a^2 ∧ a + b = Real.sqrt area_ABCD - Real.sqrt area_overlap) :
  ∃ (shaded_area : ℝ), shaded_area = 72 ∧ 
    shaded_area = area_ABCD - (((Real.sqrt area_ABCD - Real.sqrt area_overlap)/3)^2 + 4*((Real.sqrt area_ABCD - Real.sqrt area_overlap)/3)^2 - area_overlap) :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3711_371139


namespace NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l3711_371127

def cube_volume (s : ℝ) : ℝ := s^3
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem volume_of_cube_with_triple_surface_area (cube_a_side : ℝ) (cube_b_side : ℝ) :
  cube_volume cube_a_side = 8 →
  cube_surface_area cube_b_side = 3 * cube_surface_area cube_a_side →
  cube_volume cube_b_side = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l3711_371127


namespace NUMINAMATH_CALUDE_parabola_equation_and_chord_length_l3711_371132

/-- Parabola with vertex at origin, focus on positive y-axis, and focus-directrix distance 2 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex_at_origin : equation 0 0
  focus_on_y_axis : ∃ y > 0, equation 0 y
  focus_directrix_distance : ℝ

/-- Line defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y = m * x + b

theorem parabola_equation_and_chord_length 
  (p : Parabola) 
  (h_dist : p.focus_directrix_distance = 2) 
  (l : Line) 
  (h_line : l.m = 2 ∧ l.b = 1) :
  (∀ x y, p.equation x y ↔ x^2 = 4*y) ∧
  (∃ A B : ℝ × ℝ, 
    p.equation A.1 A.2 ∧ 
    p.equation B.1 B.2 ∧ 
    l.equation A.1 A.2 ∧ 
    l.equation B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_and_chord_length_l3711_371132


namespace NUMINAMATH_CALUDE_bee_swarm_count_l3711_371160

theorem bee_swarm_count : ∃ x : ℕ, 
  x > 0 ∧ 
  (x / 5 : ℚ) + (x / 3 : ℚ) + 3 * ((x / 3 : ℚ) - (x / 5 : ℚ)) + 1 = x ∧ 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bee_swarm_count_l3711_371160


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_fourteen_thirds_l3711_371184

theorem sum_abcd_equals_negative_fourteen_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) : 
  a + b + c + d = -14/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_fourteen_thirds_l3711_371184


namespace NUMINAMATH_CALUDE_initial_ducks_l3711_371166

theorem initial_ducks (initial additional total : ℕ) 
  (h1 : additional = 20)
  (h2 : total = 33)
  (h3 : initial + additional = total) : 
  initial = 13 := by
sorry

end NUMINAMATH_CALUDE_initial_ducks_l3711_371166


namespace NUMINAMATH_CALUDE_jake_not_dropping_coffee_percentage_l3711_371148

-- Define the probabilities
def trip_probability : ℝ := 0.4
def drop_coffee_when_tripping_probability : ℝ := 0.25

-- Theorem to prove
theorem jake_not_dropping_coffee_percentage :
  1 - (trip_probability * drop_coffee_when_tripping_probability) = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_jake_not_dropping_coffee_percentage_l3711_371148


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_two_l3711_371126

/-- Two vectors a and b in 2D space -/
def a : Fin 2 → ℝ := ![(-2 : ℝ), 3]
def b : ℝ → Fin 2 → ℝ := λ m => ![3, m]

/-- The dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- Theorem: If vectors a and b are perpendicular, then m = 2 -/
theorem perpendicular_vectors_m_equals_two :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_equals_two_l3711_371126


namespace NUMINAMATH_CALUDE_eriks_remaining_money_is_43_47_l3711_371114

/-- Calculates the amount of money Erik has left after his purchase --/
def eriks_remaining_money (initial_amount : ℚ) (bread_price carton_price egg_price chocolate_price : ℚ)
  (bread_quantity carton_quantity egg_quantity chocolate_quantity : ℕ)
  (discount_rate tax_rate : ℚ) : ℚ :=
  let total_cost := bread_price * bread_quantity + carton_price * carton_quantity +
                    egg_price * egg_quantity + chocolate_price * chocolate_quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  initial_amount - final_cost

/-- Theorem stating that Erik has $43.47 left after his purchase --/
theorem eriks_remaining_money_is_43_47 :
  eriks_remaining_money 86 3 6 4 2 3 3 2 5 (1/10) (1/20) = 43.47 := by
  sorry

end NUMINAMATH_CALUDE_eriks_remaining_money_is_43_47_l3711_371114


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l3711_371169

theorem opposite_of_one_half : 
  (-(1/2) : ℚ) = (-1/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l3711_371169


namespace NUMINAMATH_CALUDE_opposite_seats_theorem_l3711_371129

/-- Represents a circular seating arrangement -/
structure CircularArrangement where
  total_seats : ℕ
  is_valid : total_seats > 0

/-- Checks if two positions are opposite in a circular arrangement -/
def are_opposite (c : CircularArrangement) (pos1 pos2 : ℕ) : Prop :=
  pos1 ≤ c.total_seats ∧ pos2 ≤ c.total_seats ∧
  (pos2 - pos1) % c.total_seats = c.total_seats / 2

/-- The main theorem stating that if positions 10 and 29 are opposite, 
    the total number of seats is 38 -/
theorem opposite_seats_theorem :
  ∀ c : CircularArrangement, are_opposite c 10 29 → c.total_seats = 38 :=
by sorry

end NUMINAMATH_CALUDE_opposite_seats_theorem_l3711_371129


namespace NUMINAMATH_CALUDE_ceil_sum_sqrt_l3711_371112

theorem ceil_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 35⌉ + ⌈Real.sqrt 350⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceil_sum_sqrt_l3711_371112


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3711_371128

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The major axis length of the ellipse is 6.4 --/
theorem ellipse_major_axis_length :
  let cylinder_radius : ℝ := 2
  let major_minor_ratio : ℝ := 1.6
  major_axis_length cylinder_radius major_minor_ratio = 6.4 := by
  sorry

#eval major_axis_length 2 1.6

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3711_371128


namespace NUMINAMATH_CALUDE_range_of_inequality_l3711_371176

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def monotonic_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f x < f y

def even_function_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 2) = f (-x - 2)

-- State the theorem
theorem range_of_inequality (h1 : monotonic_increasing_on_interval f) 
                            (h2 : even_function_shifted f) :
  {x : ℝ | f (2 * x) < f (x + 2)} = Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_inequality_l3711_371176


namespace NUMINAMATH_CALUDE_sine_function_expression_l3711_371144

theorem sine_function_expression 
  (y : ℝ → ℝ) 
  (A ω : ℝ) 
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : ∀ x, y x = A * Real.sin (ω * x + φ))
  (h4 : A = 2)
  (h5 : 2 * Real.pi / ω = Real.pi / 2)
  (h6 : φ = -3) :
  ∀ x, y x = 2 * Real.sin (4 * x - 3) := by
sorry

end NUMINAMATH_CALUDE_sine_function_expression_l3711_371144


namespace NUMINAMATH_CALUDE_original_triangle_area_l3711_371174

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with area 144 square feet,
    prove that the area of the original triangle is 9 square feet. -/
theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ side, new_area = (4 * side)^2 / 2 * (original_area / (side^2 / 2))) → 
  new_area = 144 → 
  original_area = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3711_371174


namespace NUMINAMATH_CALUDE_digits_of_two_power_fifteen_times_five_power_ten_l3711_371103

theorem digits_of_two_power_fifteen_times_five_power_ten : 
  (Nat.digits 10 (2^15 * 5^10)).length = 12 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_two_power_fifteen_times_five_power_ten_l3711_371103


namespace NUMINAMATH_CALUDE_ratio_is_one_to_two_l3711_371150

/-- Represents a co-ed softball team -/
structure CoedSoftballTeam where
  men : ℕ
  women : ℕ
  total_players : ℕ
  women_more_than_men : women = men + 5
  total_is_sum : total_players = men + women

/-- The ratio of men to women in a co-ed softball team -/
def ratio_men_to_women (team : CoedSoftballTeam) : ℚ × ℚ :=
  (team.men, team.women)

theorem ratio_is_one_to_two (team : CoedSoftballTeam) 
    (h : team.total_players = 15) : 
    ratio_men_to_women team = (1, 2) := by
  sorry

#check ratio_is_one_to_two

end NUMINAMATH_CALUDE_ratio_is_one_to_two_l3711_371150


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3711_371119

/-- Given an arithmetic sequence with sum S_n = a n^2, prove a_5/d = 9/2 -/
theorem arithmetic_sequence_ratio (a : ℝ) (d : ℝ) (S : ℕ → ℝ) (a_seq : ℕ → ℝ) :
  d ≠ 0 →
  (∀ n : ℕ, S n = a * n^2) →
  (∀ n : ℕ, a_seq (n + 1) - a_seq n = d) →
  (∀ n : ℕ, S n = (n * (a_seq 1 + a_seq n)) / 2) →
  a_seq 5 / d = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3711_371119


namespace NUMINAMATH_CALUDE_total_grapes_is_83_l3711_371107

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_grapes_is_83_l3711_371107


namespace NUMINAMATH_CALUDE_fraction_doubles_when_variables_double_l3711_371162

theorem fraction_doubles_when_variables_double (x y : ℝ) (h : x + y ≠ 0) :
  (2*x)^2 / (2*x + 2*y) = 2 * (x^2 / (x + y)) :=
sorry

end NUMINAMATH_CALUDE_fraction_doubles_when_variables_double_l3711_371162


namespace NUMINAMATH_CALUDE_policy_support_percentage_l3711_371147

theorem policy_support_percentage
  (total_population : ℕ)
  (men_count : ℕ)
  (women_count : ℕ)
  (men_support_rate : ℚ)
  (women_support_rate : ℚ)
  (h1 : total_population = men_count + women_count)
  (h2 : total_population = 1000)
  (h3 : men_count = 200)
  (h4 : women_count = 800)
  (h5 : men_support_rate = 70 / 100)
  (h6 : women_support_rate = 75 / 100)
  : (men_count * men_support_rate + women_count * women_support_rate) / total_population = 74 / 100 := by
  sorry

end NUMINAMATH_CALUDE_policy_support_percentage_l3711_371147


namespace NUMINAMATH_CALUDE_parabola_properties_l3711_371115

/-- A parabola with coefficient a < 0 intersecting x-axis at (-3,0) and (1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  h_root1 : a * (-3)^2 + b * (-3) + c = 0
  h_root2 : a * 1^2 + b * 1 + c = 0

theorem parabola_properties (p : Parabola) :
  (p.b^2 - 4 * p.a * p.c > 0) ∧ (3 * p.b + 2 * p.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3711_371115


namespace NUMINAMATH_CALUDE_gcf_is_correct_l3711_371199

def term1 (x y : ℕ) : ℕ := 9 * x^3 * y^2
def term2 (x y : ℕ) : ℕ := 12 * x^2 * y^3

def gcf (x y : ℕ) : ℕ := 3 * x^2 * y^2

theorem gcf_is_correct (x y : ℕ) :
  (gcf x y) ∣ (term1 x y) ∧ (gcf x y) ∣ (term2 x y) ∧
  ∀ (d : ℕ), d ∣ (term1 x y) ∧ d ∣ (term2 x y) → d ∣ (gcf x y) :=
sorry

end NUMINAMATH_CALUDE_gcf_is_correct_l3711_371199


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l3711_371173

/-- Quadratic function g(x) = ax^2 + bx + c -/
def g (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_k_value (a b c : ℤ) :
  g a b c 2 = 0 →
  90 < g a b c 9 ∧ g a b c 9 < 100 →
  120 < g a b c 10 ∧ g a b c 10 < 130 →
  (∃ k : ℤ, 7000 * k < g a b c 150 ∧ g a b c 150 < 7000 * (k + 1)) →
  (∃ k : ℤ, 7000 * k < g a b c 150 ∧ g a b c 150 < 7000 * (k + 1) ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l3711_371173


namespace NUMINAMATH_CALUDE_wall_building_time_l3711_371168

/-- Given that 18 persons can build a 140 m long wall in 42 days, 
    prove that 30 persons will take 18 days to complete a similar 100 m long wall. -/
theorem wall_building_time 
  (original_workers : ℕ) 
  (original_length : ℝ) 
  (original_days : ℕ) 
  (new_workers : ℕ) 
  (new_length : ℝ) 
  (h1 : original_workers = 18) 
  (h2 : original_length = 140) 
  (h3 : original_days = 42) 
  (h4 : new_workers = 30) 
  (h5 : new_length = 100) :
  (new_length / new_workers) / (original_length / original_workers) * original_days = 18 :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l3711_371168


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3711_371133

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l3711_371133


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l3711_371116

/-- Given y = x^2 + 1/x^2, prove that x^6 + x^4 - 5x^3 + x^2 + 1 = 0 is equivalent to x^3(y+1) + y = -5x^3 -/
theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x^2 + 1/x^2) :
  x^6 + x^4 - 5*x^3 + x^2 + 1 = 0 ↔ x^3*(y+1) + y = -5*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l3711_371116


namespace NUMINAMATH_CALUDE_mitch_boat_financing_l3711_371156

/-- The amount Mitch has saved in dollars -/
def total_savings : ℕ := 20000

/-- The cost of a new boat per foot in dollars -/
def boat_cost_per_foot : ℕ := 1500

/-- The maximum length of boat Mitch can buy in feet -/
def max_boat_length : ℕ := 12

/-- The amount Mitch needs to keep for license and registration in dollars -/
def license_registration_cost : ℕ := 500

/-- The ratio of docking fees to license and registration cost -/
def docking_fee_ratio : ℕ := 3

theorem mitch_boat_financing :
  license_registration_cost * (docking_fee_ratio + 1) = 
    total_savings - (boat_cost_per_foot * max_boat_length) :=
by sorry

end NUMINAMATH_CALUDE_mitch_boat_financing_l3711_371156


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3711_371188

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (6 * x) + Real.sin (10 * x) = 2 * Real.sin (8 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3711_371188


namespace NUMINAMATH_CALUDE_trapezoid_area_l3711_371153

-- Define the rectangle ABCD
structure Rectangle :=
  (AB : ℝ)
  (AD : ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the trapezoid AFCB
structure Trapezoid :=
  (AB : ℝ)
  (FC : ℝ)
  (AD : ℝ)

-- Define the problem setup
def setup (rect : Rectangle) (circ : Circle) : Prop :=
  rect.AB = 32 ∧
  rect.AD = 40 ∧
  -- Circle is tangent to AB and AD
  circ.radius ≤ min rect.AB rect.AD ∧
  -- E is on BC and BE = 1
  1 ≤ rect.AB - circ.radius

-- Theorem statement
theorem trapezoid_area 
  (rect : Rectangle) 
  (circ : Circle) 
  (trap : Trapezoid) 
  (h : setup rect circ) :
  trap.AB = rect.AB ∧ 
  trap.AD = rect.AD ∧
  trap.FC = 27 → 
  (trap.AB + trap.FC) * trap.AD / 2 = 1180 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3711_371153


namespace NUMINAMATH_CALUDE_race_distance_proof_l3711_371134

/-- The distance John was behind Steve when he began his final push -/
def initial_distance : ℝ := 16

/-- John's speed in meters per second -/
def john_speed : ℝ := 4.2

/-- Steve's speed in meters per second -/
def steve_speed : ℝ := 3.7

/-- Duration of the final push in seconds -/
def final_push_duration : ℝ := 36

/-- The distance John finishes ahead of Steve -/
def final_distance_ahead : ℝ := 2

theorem race_distance_proof :
  john_speed * final_push_duration = 
  steve_speed * final_push_duration + initial_distance + final_distance_ahead :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l3711_371134


namespace NUMINAMATH_CALUDE_expansion_coefficients_l3711_371130

theorem expansion_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x : ℝ, (x + 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₁ = 7 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 127) := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l3711_371130


namespace NUMINAMATH_CALUDE_trig_system_solution_l3711_371118

theorem trig_system_solution (x y : ℝ) (m n : ℤ) :
  (Real.sin x * Real.cos y = 0.25) ∧ (Real.sin y * Real.cos x = 0.75) →
  ((x = Real.pi / 6 + Real.pi * (m - n : ℝ) ∧ y = Real.pi / 3 + Real.pi * (m + n : ℝ)) ∨
   (x = -Real.pi / 6 + Real.pi * (m - n : ℝ) ∧ y = 2 * Real.pi / 3 + Real.pi * (m + n : ℝ))) :=
by sorry

end NUMINAMATH_CALUDE_trig_system_solution_l3711_371118


namespace NUMINAMATH_CALUDE_sum_of_two_elements_equals_power_of_two_l3711_371172

def M : Set ℕ := {m : ℕ | ∃ n : ℕ, m = n * (n + 1)}

theorem sum_of_two_elements_equals_power_of_two :
  ∃ n : ℕ, n * (n - 1) ∈ M ∧ n * (n + 1) ∈ M ∧ n * (n - 1) + n * (n + 1) = 2^2021 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_elements_equals_power_of_two_l3711_371172


namespace NUMINAMATH_CALUDE_equation_solution_l3711_371104

theorem equation_solution :
  ∃ s : ℚ, (s - 60) / 3 = (6 - 3 * s) / 4 ∧ s = 258 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3711_371104


namespace NUMINAMATH_CALUDE_german_students_count_l3711_371179

theorem german_students_count (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) :
  total = 79 →
  french = 41 →
  both = 9 →
  neither = 25 →
  ∃ german : ℕ, german = 22 ∧ 
    total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_german_students_count_l3711_371179


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l3711_371192

theorem geometric_progression_equality 
  (a b q : ℝ) 
  (n p : ℕ) 
  (h_q : q ≠ 1) 
  (h_sum : a * (1 - q^(n*p)) / (1 - q) = b * (1 - q^(n*p)) / (1 - q^p)) :
  b = a * (1 - q^p) / (1 - q) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l3711_371192


namespace NUMINAMATH_CALUDE_rectangular_shape_x_value_l3711_371197

/-- A shape formed entirely of rectangles with all internal angles 90 degrees -/
structure RectangularShape where
  top_lengths : List ℝ
  bottom_lengths : List ℝ

/-- The sum of lengths in a list -/
def sum_lengths (lengths : List ℝ) : ℝ := lengths.sum

/-- The property that the sum of top lengths equals the sum of bottom lengths -/
def equal_total_length (shape : RectangularShape) : Prop :=
  sum_lengths shape.top_lengths = sum_lengths shape.bottom_lengths

theorem rectangular_shape_x_value (shape : RectangularShape) 
  (h1 : shape.top_lengths = [2, 3, 4, X])
  (h2 : shape.bottom_lengths = [1, 2, 4, 6])
  (h3 : equal_total_length shape) :
  X = 4 := by
  sorry

#check rectangular_shape_x_value

end NUMINAMATH_CALUDE_rectangular_shape_x_value_l3711_371197


namespace NUMINAMATH_CALUDE_combinations_equal_200_l3711_371140

/-- The number of varieties of gift bags -/
def gift_bags : ℕ := 10

/-- The number of colors of tissue paper -/
def tissue_papers : ℕ := 4

/-- The number of types of tags -/
def tags : ℕ := 5

/-- The total number of possible combinations -/
def total_combinations : ℕ := gift_bags * tissue_papers * tags

/-- Theorem stating that the total number of combinations is 200 -/
theorem combinations_equal_200 : total_combinations = 200 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_200_l3711_371140


namespace NUMINAMATH_CALUDE_appliance_savings_l3711_371123

def in_store_price : ℚ := 104.50
def tv_payment : ℚ := 24.80
def tv_shipping : ℚ := 10.80
def in_store_discount : ℚ := 5

theorem appliance_savings : 
  (4 * tv_payment + tv_shipping - (in_store_price - in_store_discount)) * 100 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_appliance_savings_l3711_371123


namespace NUMINAMATH_CALUDE_perpendicular_lines_minimum_value_l3711_371109

theorem perpendicular_lines_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (-(1 : ℝ) / (a - 4)) * (-2 * b) = 1) : 
  ∃ (x : ℝ), ∀ (y : ℝ), (a + 2) / (a + 1) + 1 / (2 * b) ≥ x ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
  (-(1 : ℝ) / (a₀ - 4)) * (-2 * b₀) = 1 ∧ 
  (a₀ + 2) / (a₀ + 1) + 1 / (2 * b₀) = x ∧ 
  x = 9 / 5 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_minimum_value_l3711_371109


namespace NUMINAMATH_CALUDE_circle_center_l3711_371122

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center (x y : ℝ) : 
  (3 * x + 4 * y = 24) →  -- First tangent line
  (3 * x + 4 * y = -16) →  -- Second tangent line
  (x - 2 * y = 0) →  -- Line containing the center
  (x = 4/5 ∧ y = 2/5)  -- Center coordinates
  :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3711_371122


namespace NUMINAMATH_CALUDE_remaining_boys_average_weight_l3711_371187

/-- The average weight of the remaining 8 boys given the following conditions:
  - There are 20 boys with an average weight of 50.25 kg
  - There are 8 remaining boys
  - The average weight of all 28 boys is 48.792857142857144 kg
-/
theorem remaining_boys_average_weight :
  let num_group1 : ℕ := 20
  let avg_group1 : ℝ := 50.25
  let num_group2 : ℕ := 8
  let total_num : ℕ := num_group1 + num_group2
  let total_avg : ℝ := 48.792857142857144
  
  ((num_group1 : ℝ) * avg_group1 + (num_group2 : ℝ) * avg_group2) / (total_num : ℝ) = total_avg →
  avg_group2 = 45.15
  := by sorry

end NUMINAMATH_CALUDE_remaining_boys_average_weight_l3711_371187


namespace NUMINAMATH_CALUDE_two_digit_numbers_dividing_all_relatives_l3711_371194

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_relative (ab n : ℕ) : Prop :=
  is_two_digit ab ∧
  n % 10 = ab % 10 ∧
  ∀ d ∈ (n / 10).digits 10, d ≠ 0 ∧
  digit_sum (n / 10) = ab / 10

def divides_all_relatives (ab : ℕ) : Prop :=
  is_two_digit ab ∧
  ∀ n : ℕ, is_relative ab n → ab ∣ n

theorem two_digit_numbers_dividing_all_relatives :
  {ab : ℕ | divides_all_relatives ab} =
  {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 45, 90} :=
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_dividing_all_relatives_l3711_371194


namespace NUMINAMATH_CALUDE_final_mixture_is_all_x_l3711_371155

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
structure FinalMixture where
  x : ℝ
  y : ℝ

/-- Seed mixture X -/
def X : SeedMixture :=
  { ryegrass := 1 - 0.6
    bluegrass := 0.6
    fescue := 0 }

/-- Seed mixture Y -/
def Y : SeedMixture :=
  { ryegrass := 0.25
    bluegrass := 0
    fescue := 0.75 }

/-- Theorem stating that the percentage of seed mixture X in the final mixture is 100% -/
theorem final_mixture_is_all_x (m : FinalMixture) :
  X.ryegrass * m.x + Y.ryegrass * m.y = 0.4 * (m.x + m.y) →
  m.x + m.y = 1 →
  m.x = 1 := by
  sorry


end NUMINAMATH_CALUDE_final_mixture_is_all_x_l3711_371155


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3711_371182

theorem parallelogram_side_length (s : ℝ) : 
  s > 0 → -- Ensure s is positive
  let side1 := 3 * s
  let side2 := s
  let angle := 30 * π / 180 -- Convert 30 degrees to radians
  let area := side1 * side2 * Real.sin angle
  area = 9 * Real.sqrt 3 → s = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3711_371182


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l3711_371138

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 7

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := morning_emails - email_difference

theorem jack_afternoon_emails : afternoon_emails = 3 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l3711_371138


namespace NUMINAMATH_CALUDE_sum_of_squares_mod_13_l3711_371190

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_mod_13 : sum_of_squares 15 % 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_mod_13_l3711_371190


namespace NUMINAMATH_CALUDE_fallen_striped_tiles_count_l3711_371157

/-- Represents the type of a tile -/
inductive TileType
| Striped
| Plain

/-- Represents the state of a tile position -/
inductive TileState
| Present
| Fallen

/-- Represents the initial checkerboard pattern -/
def initialPattern : List (List TileType) :=
  List.replicate 7 (List.replicate 7 TileType.Striped)

/-- Represents the current state of the wall after some tiles have fallen -/
def currentState : List (List TileState) :=
  [
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present],
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Fallen, TileState.Fallen],
    [TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Fallen],
    [TileState.Fallen, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Fallen],
    [TileState.Fallen, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Present, TileState.Present],
    [TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present],
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present]
  ]

/-- Counts the number of fallen striped tiles -/
def countFallenStripedTiles (initial : List (List TileType)) (current : List (List TileState)) : Nat :=
  sorry

/-- Theorem: The number of fallen striped tiles is 15 -/
theorem fallen_striped_tiles_count :
  countFallenStripedTiles initialPattern currentState = 15 := by
  sorry

end NUMINAMATH_CALUDE_fallen_striped_tiles_count_l3711_371157


namespace NUMINAMATH_CALUDE_erased_number_proof_l3711_371193

theorem erased_number_proof (n : ℕ) (i : ℕ) :
  n > 0 ∧ i > 0 ∧ i ≤ n ∧
  (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17 →
  i = 7 :=
by sorry

end NUMINAMATH_CALUDE_erased_number_proof_l3711_371193


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3711_371159

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + x*y + 1 ≥ C*(x + y)) ↔ C ≤ 2/Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3711_371159


namespace NUMINAMATH_CALUDE_lines_parabolas_intersection_empty_l3711_371151

-- Define the set of all lines
def Lines := {f : ℝ → ℝ | ∃ (m b : ℝ), ∀ x, f x = m * x + b}

-- Define the set of all parabolas
def Parabolas := {f : ℝ → ℝ | ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c}

-- Theorem statement
theorem lines_parabolas_intersection_empty : Lines ∩ Parabolas = ∅ := by
  sorry

end NUMINAMATH_CALUDE_lines_parabolas_intersection_empty_l3711_371151


namespace NUMINAMATH_CALUDE_max_gcd_sum_1023_l3711_371178

theorem max_gcd_sum_1023 :
  ∃ (c d : ℕ+), c + d = 1023 ∧
  ∀ (x y : ℕ+), x + y = 1023 → Nat.gcd x y ≤ Nat.gcd c d ∧
  Nat.gcd c d = 341 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1023_l3711_371178


namespace NUMINAMATH_CALUDE_unique_solution_system_l3711_371120

/-- Given positive real numbers a, b, c, prove that the unique solution to the system of equations:
    1. x + y + z = a + b + c
    2. 4xyz - (a²x + b²y + c²z) = abc
    is x = (b+c)/2, y = (c+a)/2, z = (a+b)/2, where x, y, z are positive real numbers. -/
theorem unique_solution_system (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by sorry


end NUMINAMATH_CALUDE_unique_solution_system_l3711_371120


namespace NUMINAMATH_CALUDE_tan_value_first_quadrant_l3711_371100

theorem tan_value_first_quadrant (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : Real.cos α = 2/3) : 
  Real.tan α = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_tan_value_first_quadrant_l3711_371100


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l3711_371113

theorem integer_solutions_of_system : 
  ∀ (x y z t : ℤ), 
    (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
    ((x, y, z, t) = (1, 0, 3, 1) ∨ 
     (x, y, z, t) = (-1, 0, -3, -1) ∨ 
     (x, y, z, t) = (3, 1, 1, 0) ∨ 
     (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l3711_371113


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3711_371164

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = -1/2 * a n) →
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3711_371164


namespace NUMINAMATH_CALUDE_cube_difference_formula_l3711_371152

theorem cube_difference_formula (n : ℕ) : 
  (n + 1)^3 - n^3 = 3*n^2 + 3*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_formula_l3711_371152


namespace NUMINAMATH_CALUDE_program_output_l3711_371186

theorem program_output : 
  let initial_value := 2
  let after_multiplication := initial_value * 2
  let final_value := after_multiplication + 6
  final_value = 10 := by sorry

end NUMINAMATH_CALUDE_program_output_l3711_371186


namespace NUMINAMATH_CALUDE_power_of_64_l3711_371175

theorem power_of_64 : 64^(5/3) = 1024 := by
  have h : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l3711_371175
