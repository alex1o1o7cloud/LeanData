import Mathlib

namespace A_suff_not_nec_D_l505_50581

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_and_suff_C : (B ↔ C)
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem A_suff_not_nec_D : (A → D) ∧ ¬(D → A) :=
sorry

end A_suff_not_nec_D_l505_50581


namespace smallest_number_with_conditions_l505_50591

def containsOnly3And4 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 3 ∨ d = 4

def contains3And4 (n : ℕ) : Prop :=
  3 ∈ n.digits 10 ∧ 4 ∈ n.digits 10

def isMultipleOf3And4 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0

theorem smallest_number_with_conditions :
  ∀ n : ℕ, 
    containsOnly3And4 n ∧ 
    contains3And4 n ∧ 
    isMultipleOf3And4 n →
    n ≥ 3444 :=
by sorry

end smallest_number_with_conditions_l505_50591


namespace twentieth_term_of_sequence_l505_50599

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 20th term of the arithmetic sequence 8, 5, 2, ... -/
theorem twentieth_term_of_sequence : arithmeticSequence 8 (-3) 20 = -49 := by
  sorry

end twentieth_term_of_sequence_l505_50599


namespace scientific_notation_502000_l505_50560

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_502000 :
  toScientificNotation 502000 = ScientificNotation.mk 5.02 5 (by sorry) :=
sorry

end scientific_notation_502000_l505_50560


namespace playground_students_l505_50511

/-- The number of students initially on the playground -/
def initial_students : ℕ := 32

/-- The number of students who left the playground -/
def students_left : ℕ := 16

/-- The number of new students who came to the playground -/
def new_students : ℕ := 9

/-- The final number of students on the playground -/
def final_students : ℕ := 25

theorem playground_students :
  initial_students - students_left + new_students = final_students :=
by sorry

end playground_students_l505_50511


namespace lcm_gcf_problem_l505_50577

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 12 = 48 → Nat.gcd n 12 = 8 → n = 32 := by
  sorry

end lcm_gcf_problem_l505_50577


namespace remaining_money_l505_50566

def initial_amount : ℕ := 91
def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11

theorem remaining_money : 
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end remaining_money_l505_50566


namespace terrell_hike_distance_l505_50572

theorem terrell_hike_distance (saturday_distance sunday_distance : ℝ) 
  (h1 : saturday_distance = 8.2)
  (h2 : sunday_distance = 1.6) : 
  saturday_distance + sunday_distance = 9.8 := by
  sorry

end terrell_hike_distance_l505_50572


namespace min_vector_difference_l505_50534

/-- Given planar vectors a, b, c satisfying the conditions, 
    the minimum value of |a - b| is 6 -/
theorem min_vector_difference (a b c : ℝ × ℝ) 
    (h1 : a • b = 0)
    (h2 : ‖c‖ = 1)
    (h3 : ‖a - c‖ = 5)
    (h4 : ‖b - c‖ = 5) :
    6 ≤ ‖a - b‖ ∧ ∃ (a' b' c' : ℝ × ℝ), 
      a' • b' = 0 ∧ 
      ‖c'‖ = 1 ∧ 
      ‖a' - c'‖ = 5 ∧ 
      ‖b' - c'‖ = 5 ∧
      ‖a' - b'‖ = 6 :=
by sorry

end min_vector_difference_l505_50534


namespace equation_equivalence_l505_50587

theorem equation_equivalence :
  ∀ x : ℝ, (x - 1) / 0.2 - x / 0.5 = 1 ↔ 3 * x = 6 := by sorry

end equation_equivalence_l505_50587


namespace total_practice_time_is_135_l505_50501

/-- The number of minutes Daniel practices on a school day -/
def school_day_practice : ℕ := 15

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The number of minutes Daniel practices on a weekend day -/
def weekend_day_practice : ℕ := 2 * school_day_practice

/-- The total practice time for a whole week in minutes -/
def total_practice_time : ℕ := school_day_practice * school_days + weekend_day_practice * weekend_days

theorem total_practice_time_is_135 : total_practice_time = 135 := by
  sorry

end total_practice_time_is_135_l505_50501


namespace lcm_and_sum_first_ten_l505_50509

/-- The set of the first ten positive integers -/
def firstTenIntegers : Finset ℕ := Finset.range 10

/-- The least common multiple of the first ten positive integers -/
def lcmFirstTen : ℕ := Finset.lcm firstTenIntegers id

/-- The sum of the first ten positive integers -/
def sumFirstTen : ℕ := Finset.sum firstTenIntegers id

theorem lcm_and_sum_first_ten :
  lcmFirstTen = 2520 ∧ sumFirstTen = 55 := by sorry

end lcm_and_sum_first_ten_l505_50509


namespace room_length_is_twenty_l505_50585

/-- Represents the dimensions and tiling of a rectangular room. -/
structure Room where
  length : ℝ
  breadth : ℝ
  tileSize : ℝ
  blackTileWidth : ℝ
  blueTileCount : ℕ

/-- Theorem stating the length of the room given specific conditions. -/
theorem room_length_is_twenty (r : Room) : 
  r.breadth = 10 ∧ 
  r.tileSize = 2 ∧ 
  r.blackTileWidth = 2 ∧ 
  r.blueTileCount = 16 ∧
  (r.length - 2 * r.blackTileWidth) * (r.breadth - 2 * r.blackTileWidth) * (2/3) = 
    (r.blueTileCount : ℝ) * r.tileSize * r.tileSize →
  r.length = 20 := by
  sorry

#check room_length_is_twenty

end room_length_is_twenty_l505_50585


namespace power_of_power_l505_50595

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l505_50595


namespace sin_negative_thirty_degrees_l505_50530

theorem sin_negative_thirty_degrees : Real.sin (-(30 * π / 180)) = -1/2 := by
  sorry

end sin_negative_thirty_degrees_l505_50530


namespace line_at_distance_iff_tangent_to_cylinder_l505_50545

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  a : Point3D
  b : Point3D

/-- A cylinder in 3D space defined by its axis (a line) and radius -/
structure Cylinder where
  axis : Line3D
  radius : ℝ

/-- Distance between a point and a line in 3D space -/
def distance_point_to_line (p : Point3D) (l : Line3D) : ℝ := sorry

/-- Check if a line is tangent to a cylinder -/
def is_tangent_to_cylinder (l : Line3D) (c : Cylinder) : Prop := sorry

/-- Check if a line passes through a point -/
def line_passes_through_point (l : Line3D) (p : Point3D) : Prop := sorry

/-- Main theorem: A line passing through M is at distance d from AB iff it's tangent to the cylinder -/
theorem line_at_distance_iff_tangent_to_cylinder 
  (M : Point3D) (AB : Line3D) (d : ℝ) (l : Line3D) : 
  (line_passes_through_point l M ∧ distance_point_to_line M AB = d) ↔ 
  is_tangent_to_cylinder l (Cylinder.mk AB d) :=
sorry

end line_at_distance_iff_tangent_to_cylinder_l505_50545


namespace square_sum_of_product_and_sum_l505_50552

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end square_sum_of_product_and_sum_l505_50552


namespace fair_haired_women_percentage_l505_50516

theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 28)
  (h2 : fair_hair_percentage = 70) :
  (women_fair_hair_percentage / fair_hair_percentage) * 100 = 40 := by
sorry

end fair_haired_women_percentage_l505_50516


namespace distance_to_symmetry_axis_range_l505_50542

variable (a b c : ℝ)
variable (f : ℝ → ℝ)

theorem distance_to_symmetry_axis_range 
  (ha : a > 0)
  (hf : f = fun x ↦ a * x^2 + b * x + c)
  (htangent : ∀ x₀, 0 ≤ (2 * a * x₀ + b) ∧ (2 * a * x₀ + b) ≤ 1) :
  ∃ d : Set ℝ, d = {x | 0 ≤ x ∧ x ≤ 1 / (2 * a)} ∧
    ∀ x₀, Set.Mem (|x₀ + b / (2 * a)|) d :=
by sorry

end distance_to_symmetry_axis_range_l505_50542


namespace balloon_arrangements_l505_50594

theorem balloon_arrangements (n : ℕ) (n1 n2 n3 n4 n5 : ℕ) : 
  n = 7 → 
  n1 = 2 → 
  n2 = 2 → 
  n3 = 1 → 
  n4 = 1 → 
  n5 = 1 → 
  (n.factorial) / (n1.factorial * n2.factorial * n3.factorial * n4.factorial * n5.factorial) = 1260 := by
  sorry

end balloon_arrangements_l505_50594


namespace factorization_equality_l505_50506

theorem factorization_equality (m x : ℝ) : m * x^2 - 4 * m = m * (x + 2) * (x - 2) := by
  sorry

end factorization_equality_l505_50506


namespace f_five_equals_142_l505_50559

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_five_equals_142 :
  ∃ y : ℝ, (f 2 y = 100) ∧ (f 5 y = 142) := by
  sorry

end f_five_equals_142_l505_50559


namespace parallelogram_area_l505_50584

theorem parallelogram_area (base height : ℝ) (h1 : base = 12) (h2 : height = 18) :
  base * height = 216 :=
sorry

end parallelogram_area_l505_50584


namespace celeste_candy_theorem_l505_50580

/-- Represents the state of candies on the table -/
structure CandyState (n : ℕ+) where
  counts : Fin n → ℕ

/-- Represents the operations that can be performed on the candy state -/
inductive Operation (n : ℕ+)
  | split : Fin n → Operation n
  | take : Fin n → Operation n

/-- Applies an operation to a candy state -/
def apply_operation {n : ℕ+} (state : CandyState n) (op : Operation n) : CandyState n :=
  sorry

/-- Checks if a candy state is empty -/
def is_empty {n : ℕ+} (state : CandyState n) : Prop :=
  ∀ i, state.counts i = 0

/-- Main theorem: Celeste can empty the table for any initial configuration
    if and only if n is not divisible by 3 -/
theorem celeste_candy_theorem (n : ℕ+) :
  (∀ (m : ℕ+) (initial_state : CandyState n),
    ∃ (ops : List (Operation n)), is_empty (ops.foldl apply_operation initial_state))
  ↔ ¬(n : ℕ) % 3 = 0 := by
  sorry

end celeste_candy_theorem_l505_50580


namespace chromium_content_bounds_l505_50525

/-- Represents the chromium content in an alloy mixture -/
structure ChromiumAlloy where
  x : ℝ  -- Relative mass of 1st alloy
  y : ℝ  -- Relative mass of 2nd alloy
  z : ℝ  -- Relative mass of 3rd alloy
  k : ℝ  -- Chromium content

/-- Conditions for a valid ChromiumAlloy -/
def is_valid_alloy (a : ChromiumAlloy) : Prop :=
  a.x ≥ 0 ∧ a.y ≥ 0 ∧ a.z ≥ 0 ∧
  a.x + a.y + a.z = 1 ∧
  0.9 * a.x + 0.3 * a.z = 0.45 ∧
  0.4 * a.x + 0.1 * a.y + 0.5 * a.z = a.k

theorem chromium_content_bounds (a : ChromiumAlloy) 
  (h : is_valid_alloy a) : 
  a.k ≥ 0.25 ∧ a.k ≤ 0.4 := by
  sorry

end chromium_content_bounds_l505_50525


namespace f_properties_l505_50554

def f (x : ℝ) := x^3 - 3*x^2 + 6

theorem f_properties :
  (∃ (a : ℝ), IsLocalMin f a ∧ f a = 2) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 6) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 6) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, 2 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 2) := by
  sorry


end f_properties_l505_50554


namespace sum_product_range_l505_50550

theorem sum_product_range (a b c : ℝ) (h : a + b + c = 0) :
  (∀ x : ℝ, x ≤ 0 → ∃ a b c : ℝ, a + b + c = 0 ∧ a * b + a * c + b * c = x) ∧
  (∀ a b c : ℝ, a + b + c = 0 → a * b + a * c + b * c ≤ 0) :=
by sorry

end sum_product_range_l505_50550


namespace f_value_at_5pi_3_l505_50570

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

theorem f_value_at_5pi_3 (f : ℝ → ℝ) (h1 : is_even f)
  (h2 : smallest_positive_period f π)
  (h3 : ∀ x ∈ Set.Icc 0 (π/2), f x = Real.cos x) :
  f (5*π/3) = 1/2 := by
  sorry

end f_value_at_5pi_3_l505_50570


namespace cat_food_cans_l505_50543

/-- The number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := sorry

/-- The number of packages of cat food -/
def cat_packages : ℕ := 6

/-- The number of packages of dog food -/
def dog_packages : ℕ := 2

/-- The number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 3

theorem cat_food_cans : 
  cat_packages * cans_per_cat_package = 
  dog_packages * cans_per_dog_package + 48 ∧ 
  cans_per_cat_package = 9 :=
sorry

end cat_food_cans_l505_50543


namespace sphere_volume_equals_cone_cylinder_volume_l505_50558

/-- Given a cone with height 6 and radius 1.5, and a cylinder with the same height and volume as the cone,
    prove that a sphere with radius 1.5 has the same volume as both the cone and cylinder. -/
theorem sphere_volume_equals_cone_cylinder_volume :
  let cone_height : ℝ := 6
  let cone_radius : ℝ := 1.5
  let cylinder_height : ℝ := cone_height
  let cone_volume : ℝ := (1 / 3) * Real.pi * cone_radius^2 * cone_height
  let cylinder_volume : ℝ := cone_volume
  let sphere_radius : ℝ := 1.5
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = cone_volume := by
  sorry


end sphere_volume_equals_cone_cylinder_volume_l505_50558


namespace min_value_xy_expression_min_value_achievable_l505_50520

theorem min_value_xy_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by sorry

end min_value_xy_expression_min_value_achievable_l505_50520


namespace original_number_proof_l505_50518

theorem original_number_proof : ∃! n : ℤ, n * 74 = 19732 := by
  sorry

end original_number_proof_l505_50518


namespace white_then_red_probability_l505_50553

/-- The probability of drawing a white marble first and a red marble second from a bag with 4 red and 6 white marbles -/
theorem white_then_red_probability : 
  let total_marbles : ℕ := 4 + 6
  let red_marbles : ℕ := 4
  let white_marbles : ℕ := 6
  let prob_white_first : ℚ := white_marbles / total_marbles
  let prob_red_second : ℚ := red_marbles / (total_marbles - 1)
  prob_white_first * prob_red_second = 4 / 15 :=
by sorry

end white_then_red_probability_l505_50553


namespace total_carrots_grown_l505_50524

theorem total_carrots_grown (sandy_carrots sam_carrots : ℕ) 
  (h1 : sandy_carrots = 6) 
  (h2 : sam_carrots = 3) : 
  sandy_carrots + sam_carrots = 9 := by
  sorry

end total_carrots_grown_l505_50524


namespace ellipse_triangle_area_l505_50592

/-- An ellipse with given properties -/
structure Ellipse :=
  (A B E F : ℝ × ℝ)
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4)
  (AF_length : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = 2 + Real.sqrt 3)

/-- A point on the ellipse satisfying the given condition -/
def PointOnEllipse (Γ : Ellipse) (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
  Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2) = 2

/-- The theorem to be proved -/
theorem ellipse_triangle_area (Γ : Ellipse) (P : ℝ × ℝ) (h : PointOnEllipse Γ P) :
  (1/2) * Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
         Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2) *
         Real.sin (Real.arccos (
           ((P.1 - Γ.E.1) * (P.1 - Γ.F.1) + (P.2 - Γ.E.2) * (P.2 - Γ.F.2)) /
           (Real.sqrt ((P.1 - Γ.E.1)^2 + (P.2 - Γ.E.2)^2) *
            Real.sqrt ((P.1 - Γ.F.1)^2 + (P.2 - Γ.F.2)^2))
         )) = 1 := by
  sorry

end ellipse_triangle_area_l505_50592


namespace line_parabola_intersection_length_l505_50521

/-- Given a line y = kx - k intersecting the parabola y^2 = 4x at points A and B,
    if the midpoint of AB is 3 units from the y-axis, then the length of AB is 8 units. -/
theorem line_parabola_intersection_length (k : ℝ) (A B : ℝ × ℝ) : 
  (∀ x y, y = k * x - k → y^2 = 4 * x → (x, y) = A ∨ (x, y) = B) →  -- Line intersects parabola at A and B
  ((A.1 + B.1) / 2 = 3) →                                           -- Midpoint is 3 units from y-axis
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=                  -- Length of AB is 8 units
by sorry

end line_parabola_intersection_length_l505_50521


namespace vector_properties_l505_50514

/-- Given two vectors in ℝ², prove dot product and parallelism properties -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (1, -2)) (h2 : b = (-3, 2)) :
  (a + b) • (a - b) = -8 ∧
  ∃ k : ℝ, k = (1 : ℝ) / 3 ∧ ∃ c : ℝ, c ≠ 0 ∧ k • a + b = c • (a - 3 • b) := by
  sorry

end vector_properties_l505_50514


namespace forty_percent_of_number_l505_50573

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (40/100 : ℝ) * N = 192 := by
  sorry

end forty_percent_of_number_l505_50573


namespace empty_seats_count_l505_50557

structure Section where
  capacity : Nat
  attendance : Nat

def theater : List Section := [
  { capacity := 250, attendance := 195 },
  { capacity := 180, attendance := 143 },
  { capacity := 150, attendance := 110 },
  { capacity := 300, attendance := 261 },
  { capacity := 230, attendance := 157 },
  { capacity := 90, attendance := 66 }
]

def totalCapacity : Nat := List.foldl (fun acc s => acc + s.capacity) 0 theater
def totalAttendance : Nat := List.foldl (fun acc s => acc + s.attendance) 0 theater

theorem empty_seats_count :
  totalCapacity - totalAttendance = 268 :=
by sorry

end empty_seats_count_l505_50557


namespace jordans_garden_area_l505_50522

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the garden in square yards --/
def garden_area (g : Garden) : ℕ :=
  (g.short_side_posts - 1) * g.post_spacing * ((g.long_side_posts - 1) * g.post_spacing)

/-- Theorem stating the area of Jordan's garden --/
theorem jordans_garden_area :
  ∀ g : Garden,
    g.total_posts = 28 →
    g.post_spacing = 3 →
    g.long_side_posts = 2 * g.short_side_posts + 3 →
    garden_area g = 630 := by
  sorry

end jordans_garden_area_l505_50522


namespace max_sections_five_lines_l505_50548

/-- The number of sections created by n line segments in a rectangle -/
def sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else sections (n - 1) + (n - 1)

/-- The maximum number of sections created by 5 line segments in a rectangle -/
theorem max_sections_five_lines :
  sections 5 = 12 :=
sorry

end max_sections_five_lines_l505_50548


namespace complex_power_sum_l505_50504

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_sum : 2 * (i^13 + i^18 + i^23 + i^28 + i^33) = 2 * i := by
  sorry

end complex_power_sum_l505_50504


namespace other_sales_percentage_l505_50544

/-- The Paper Boutique's sales percentages -/
structure SalesPercentages where
  pens : ℝ
  pencils : ℝ
  notebooks : ℝ
  total : ℝ
  pens_percent : pens = 25
  pencils_percent : pencils = 30
  notebooks_percent : notebooks = 20
  total_sum : total = 100

/-- Theorem: The percentage of sales that are neither pens, pencils, nor notebooks is 25% -/
theorem other_sales_percentage (s : SalesPercentages) : 
  s.total - (s.pens + s.pencils + s.notebooks) = 25 := by
  sorry

end other_sales_percentage_l505_50544


namespace inscribed_cube_volume_l505_50597

/-- A pyramid with a regular hexagonal base and isosceles triangular lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_base_length : ℝ
  lateral_face_height : ℝ

/-- A cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  edge_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.edge_length ^ 3

/-- Conditions for the specific pyramid and inscribed cube -/
def specific_pyramid_and_cube : Prop :=
  ∃ (p : HexagonalPyramid) (c : InscribedCube),
    p.base_side_length = 1 ∧
    p.lateral_face_base_length = 1 ∧
    c.pyramid = p ∧
    c.edge_length = 1

/-- Theorem stating that the volume of the inscribed cube is 1 -/
theorem inscribed_cube_volume :
  specific_pyramid_and_cube →
  ∃ (c : InscribedCube), cube_volume c = 1 :=
sorry

end inscribed_cube_volume_l505_50597


namespace wire_cut_square_octagon_ratio_l505_50596

theorem wire_cut_square_octagon_ratio (a b : ℝ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) :
  (a^2 / 16 = b^2 * (1 + Real.sqrt 2) / 32) → a / b = Real.sqrt ((2 + Real.sqrt 2) / 2) := by
  sorry

end wire_cut_square_octagon_ratio_l505_50596


namespace closest_integer_to_cube_root_1728_l505_50505

theorem closest_integer_to_cube_root_1728 : 
  ∃ n : ℤ, ∀ m : ℤ, |n - (1728 : ℝ)^(1/3)| ≤ |m - (1728 : ℝ)^(1/3)| ∧ n = 12 :=
by
  sorry

end closest_integer_to_cube_root_1728_l505_50505


namespace periodic_even_function_value_l505_50529

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_even_function_value 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_even : is_even f)
  (h_periodic : has_period f 6)
  (h_interval : ∀ x ∈ Set.Icc (-3) 3, f x = (x + 1) * (x - a)) :
  f (-6) = -1 :=
sorry

end periodic_even_function_value_l505_50529


namespace initial_amount_of_liquid_A_l505_50583

/-- Given a mixture of liquids A and B with an initial ratio of 4:1, prove that the initial amount
of liquid A is 16 liters when 10 L of the mixture is replaced with liquid B, resulting in a new
ratio of 2:3. -/
theorem initial_amount_of_liquid_A (x : ℝ) : 
  (4 * x) / x = 4 / 1 →  -- Initial ratio of A to B is 4:1
  ((4 * x - 8) / (x + 8) = 2 / 3) →  -- New ratio after replacement is 2:3
  4 * x = 16 :=  -- Initial amount of liquid A is 16 liters
by sorry

#check initial_amount_of_liquid_A

end initial_amount_of_liquid_A_l505_50583


namespace sum_is_negative_l505_50538

theorem sum_is_negative (x y : ℝ) (hx : x > 0) (hy : y < 0) (hxy : |x| < |y|) : x + y < 0 := by
  sorry

end sum_is_negative_l505_50538


namespace perpendicular_lines_parallel_perpendicular_to_parallel_planes_l505_50532

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perpendicular_lines_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) (h2 : perpendicular n α) : 
  parallel m n :=
sorry

-- Theorem 2: If three planes are parallel and a line is perpendicular to one of them, 
-- then it is perpendicular to all of them
theorem perpendicular_to_parallel_planes 
  (m : Line) (α β γ : Plane)
  (h1 : plane_parallel α β) (h2 : plane_parallel β γ) 
  (h3 : perpendicular m α) :
  perpendicular m γ :=
sorry

end perpendicular_lines_parallel_perpendicular_to_parallel_planes_l505_50532


namespace handshake_count_l505_50527

theorem handshake_count (n : ℕ) (h : n = 8) :
  let pairs := n / 2
  let handshakes_per_person := n - 2
  (n * handshakes_per_person) / 2 = 24 :=
by sorry

end handshake_count_l505_50527


namespace sqrt_equation_solution_l505_50549

theorem sqrt_equation_solution :
  ∃ s : ℝ, (Real.sqrt (3 * Real.sqrt (s - 1)) = (9 - s) ^ (1/4)) ∧ s = 1.8 := by
  sorry

end sqrt_equation_solution_l505_50549


namespace total_employees_after_increase_l505_50533

-- Define the initial conditions
def initial_total : ℕ := 1200
def initial_production : ℕ := 800
def initial_admin : ℕ := 400
def production_increase : ℚ := 35 / 100
def admin_increase : ℚ := 3 / 5

-- Define the theorem
theorem total_employees_after_increase : 
  initial_production * (1 + production_increase) + initial_admin * (1 + admin_increase) = 1720 := by
  sorry

end total_employees_after_increase_l505_50533


namespace hyperbola_n_range_l505_50564

/-- Represents a hyperbola with parameters m and n -/
structure Hyperbola (m n : ℝ) where
  equation : ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

/-- The distance between the foci of a hyperbola -/
def focal_distance (h : Hyperbola m n) : ℝ := 4

/-- Theorem stating the range of n for a hyperbola with given properties -/
theorem hyperbola_n_range (m n : ℝ) (h : Hyperbola m n) :
  focal_distance h = 4 → -1 < n ∧ n < 3 := by
  sorry

end hyperbola_n_range_l505_50564


namespace c_range_l505_50567

def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem c_range (a b c : ℝ) :
  (0 < f a b c (-1) ∧ f a b c (-1) = f a b c (-2) ∧ f a b c (-2) = f a b c (-3) ∧ f a b c (-3) ≤ 3) →
  (6 < c ∧ c ≤ 9) := by
  sorry

end c_range_l505_50567


namespace min_value_quadratic_min_value_achievable_l505_50561

theorem min_value_quadratic (x y : ℝ) : x^2 + y^2 - 8*x + 6*y + x*y + 20 ≥ -88/3 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + y^2 - 8*x + 6*y + x*y + 20 = -88/3 := by
  sorry

end min_value_quadratic_min_value_achievable_l505_50561


namespace conditional_probability_B_given_A_l505_50523

-- Define the set of numbers
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define the event A: "The product of the two chosen numbers is even"
def event_A (x y : ℕ) : Prop := x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ Even (x * y)

-- Define the event B: "Both chosen numbers are even"
def event_B (x y : ℕ) : Prop := x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ Even x ∧ Even y

-- Define the probability of choosing two different numbers from S
def prob_total : ℚ := 10 / 1

-- Define the probability of event A
def prob_A : ℚ := 7 / 10

-- Define the probability of event A ∩ B
def prob_A_and_B : ℚ := 1 / 10

-- Theorem statement
theorem conditional_probability_B_given_A :
  prob_A_and_B / prob_A = 1 / 7 :=
sorry

end conditional_probability_B_given_A_l505_50523


namespace positions_after_631_moves_l505_50575

/-- Represents the possible positions of the dog on the hexagon -/
inductive DogPosition
  | Top
  | TopRight
  | BottomRight
  | Bottom
  | BottomLeft
  | TopLeft

/-- Represents the possible positions of the rabbit on the hexagon -/
inductive RabbitPosition
  | TopCenter
  | TopRight
  | RightUpper
  | RightLower
  | BottomRight
  | BottomCenter
  | BottomLeft
  | LeftLower
  | LeftUpper
  | TopLeft
  | LeftCenter
  | RightCenter

/-- Calculates the position of the dog after a given number of moves -/
def dogPositionAfterMoves (moves : Nat) : DogPosition :=
  match moves % 6 with
  | 0 => DogPosition.TopLeft
  | 1 => DogPosition.Top
  | 2 => DogPosition.TopRight
  | 3 => DogPosition.BottomRight
  | 4 => DogPosition.Bottom
  | 5 => DogPosition.BottomLeft
  | _ => DogPosition.Top  -- This case is unreachable, but needed for exhaustiveness

/-- Calculates the position of the rabbit after a given number of moves -/
def rabbitPositionAfterMoves (moves : Nat) : RabbitPosition :=
  match moves % 12 with
  | 0 => RabbitPosition.RightCenter
  | 1 => RabbitPosition.TopCenter
  | 2 => RabbitPosition.TopRight
  | 3 => RabbitPosition.RightUpper
  | 4 => RabbitPosition.RightLower
  | 5 => RabbitPosition.BottomRight
  | 6 => RabbitPosition.BottomCenter
  | 7 => RabbitPosition.BottomLeft
  | 8 => RabbitPosition.LeftLower
  | 9 => RabbitPosition.LeftUpper
  | 10 => RabbitPosition.TopLeft
  | 11 => RabbitPosition.LeftCenter
  | _ => RabbitPosition.TopCenter  -- This case is unreachable, but needed for exhaustiveness

theorem positions_after_631_moves :
  dogPositionAfterMoves 631 = DogPosition.Top ∧
  rabbitPositionAfterMoves 631 = RabbitPosition.BottomLeft :=
by sorry

end positions_after_631_moves_l505_50575


namespace fixed_point_on_curve_circle_center_on_line_l505_50526

-- Define the curve C
def C (a x y : ℝ) : Prop := x^2 + y^2 - 4*a*x + 2*a*y - 20 + 20*a = 0

-- Theorem 1: The point (4, -2) always lies on C for any value of a
theorem fixed_point_on_curve (a : ℝ) : C a 4 (-2) := by sorry

-- Theorem 2: When a ≠ 2, C is a circle and its center lies on the line x + 2y = 0
theorem circle_center_on_line (a : ℝ) (h : a ≠ 2) :
  ∃ (x y : ℝ), C a x y ∧ (∀ (x' y' : ℝ), C a x' y' → (x' - x)^2 + (y' - y)^2 = (x - x')^2 + (y - y')^2) ∧ x + 2*y = 0 := by sorry

end fixed_point_on_curve_circle_center_on_line_l505_50526


namespace smallest_number_l505_50539

theorem smallest_number : ∀ (a b c d : ℚ), 
  a = 1 → b = -2 → c = 0 → d = -1/2 → 
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end smallest_number_l505_50539


namespace E_80_l505_50508

/-- E(n) represents the number of ways to express n as a product of integers greater than 1, where order matters -/
def E (n : ℕ) : ℕ := sorry

/-- The prime factorization of 80 is 2^4 * 5 -/
axiom prime_factorization_80 : 80 = 2^4 * 5

/-- Theorem: The number of ways to express 80 as a product of integers greater than 1, where order matters, is 42 -/
theorem E_80 : E 80 = 42 := by sorry

end E_80_l505_50508


namespace linear_function_properties_l505_50571

/-- Linear function defined as f(x) = -2x + 4 -/
def f (x : ℝ) : ℝ := -2 * x + 4

theorem linear_function_properties :
  /- Property 1: For any two points on the graph, if x₁ < x₂, then f(x₁) > f(x₂) -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  /- Property 2: The graph does not pass through the third quadrant -/
  (∀ x y : ℝ, f x = y → (x ≤ 0 → y ≥ 0) ∧ (y ≤ 0 → x ≥ 0)) ∧
  /- Property 3: Shifting the graph down by 4 units results in y = -2x -/
  (∀ x : ℝ, f x - 4 = -2 * x) ∧
  /- Property 4: The x-intercept is at (2, 0) -/
  (f 2 = 0 ∧ ∀ x : ℝ, f x = 0 → x = 2) :=
by sorry

end linear_function_properties_l505_50571


namespace product_zero_l505_50515

/-- Given two real numbers x and y satisfying x - y = 6 and x³ - y³ = 162, their product xy equals 0. -/
theorem product_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 := by
  sorry

end product_zero_l505_50515


namespace rice_cost_l505_50519

/-- Proves that the cost of each kilogram of rice is $2 given the conditions of Vicente's purchase --/
theorem rice_cost (rice_kg : ℕ) (meat_lb : ℕ) (meat_cost_per_lb : ℕ) (total_spent : ℕ) : 
  rice_kg = 5 → meat_lb = 3 → meat_cost_per_lb = 5 → total_spent = 25 →
  ∃ (rice_cost_per_kg : ℕ), rice_cost_per_kg = 2 ∧ rice_kg * rice_cost_per_kg + meat_lb * meat_cost_per_lb = total_spent :=
by sorry

end rice_cost_l505_50519


namespace equation_solution_l505_50574

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 2 → (x - 8 / (x - 2) = 5 + 8 / (x - 2)) ↔ (x = 9 ∨ x = -2) :=
by sorry

end equation_solution_l505_50574


namespace x_value_l505_50507

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.15 * 1600 - 15) ∧ (x = 900) := by
  sorry

end x_value_l505_50507


namespace calculation_proof_l505_50562

theorem calculation_proof : ((0.15 * 320 + 0.12 * 480) / (2/5)) * (3/4) = 198 := by
  sorry

end calculation_proof_l505_50562


namespace complex_fraction_equals_i_l505_50598

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by
  sorry

end complex_fraction_equals_i_l505_50598


namespace triangle_side_ratio_range_l505_50582

theorem triangle_side_ratio_range (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  S > 0 →
  2 * S = a^2 - (b - c)^2 →
  3 / 5 < b / c ∧ b / c < 5 / 3 := by
sorry


end triangle_side_ratio_range_l505_50582


namespace hyperbola_asymptotes_l505_50551

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 (a > 0) and eccentricity √3,
    prove that its asymptotes are y = ±√2 x -/
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) :
  let hyperbola := λ (x y : ℝ) => x^2 / a^2 - y^2 / 2 = 1
  let eccentricity := Real.sqrt 3
  let asymptotes := λ (x y : ℝ) => y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x
  ∀ (x y : ℝ), hyperbola x y ∧ eccentricity = Real.sqrt 3 → asymptotes x y :=
by sorry

end hyperbola_asymptotes_l505_50551


namespace triangle_properties_l505_50563

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) * Real.cos t.C + t.c * Real.cos t.B = 0

theorem triangle_properties (t : Triangle) 
  (h : condition t) : 
  t.C = 2 * Real.pi / 3 ∧ 
  (t.c = 6 → ∃ (max_area : ℝ), max_area = 3 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area ≤ max_area) :=
sorry

end triangle_properties_l505_50563


namespace baking_scoops_l505_50578

/-- Calculates the number of scoops needed given the amount of ingredient in cups and the size of the scoop -/
def scoops_needed (cups : ℚ) (scoop_size : ℚ) : ℕ :=
  (cups / scoop_size).ceil.toNat

/-- The total number of scoops needed for flour and sugar -/
def total_scoops : ℕ :=
  scoops_needed 3 (1/3) + scoops_needed 2 (1/3)

theorem baking_scoops : total_scoops = 15 := by
  sorry

end baking_scoops_l505_50578


namespace marble_problem_l505_50568

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 3

/-- The probability that all 3 girls select the same colored marble -/
def same_color_prob : ℚ := 1/10

/-- The number of black marbles in the bag -/
def black_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := white_marbles + black_marbles

/-- The probability of all girls selecting white marbles -/
def all_white_prob : ℚ := (white_marbles / total_marbles) * 
                          ((white_marbles - 1) / (total_marbles - 1)) * 
                          ((white_marbles - 2) / (total_marbles - 2))

/-- The probability of all girls selecting black marbles -/
def all_black_prob : ℚ := (black_marbles / total_marbles) * 
                          ((black_marbles - 1) / (total_marbles - 1)) * 
                          ((black_marbles - 2) / (total_marbles - 2))

theorem marble_problem : 
  all_white_prob + all_black_prob = same_color_prob :=
by sorry

end marble_problem_l505_50568


namespace fifth_element_row_20_l505_50547

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The fifth element in a row of Pascal's triangle -/
def fifthElement (row : ℕ) : ℕ := binomial row 4

theorem fifth_element_row_20 : fifthElement 20 = 4845 := by
  sorry

end fifth_element_row_20_l505_50547


namespace triangle_max_perimeter_l505_50555

theorem triangle_max_perimeter : 
  ∀ x y z : ℕ,
  x = 4 * y →
  z = 20 →
  (x + y > z ∧ x + z > y ∧ y + z > x) →
  ∀ a b c : ℕ,
  a = 4 * b →
  c = 20 →
  (a + b > c ∧ a + c > b ∧ b + c > a) →
  x + y + z ≤ a + b + c →
  x + y + z ≤ 50 :=
by sorry

end triangle_max_perimeter_l505_50555


namespace average_score_is_correct_l505_50531

def total_students : ℕ := 120

-- Define the score distribution
def score_distribution : List (ℕ × ℕ) := [
  (95, 12),
  (85, 24),
  (75, 30),
  (65, 20),
  (55, 18),
  (45, 10),
  (35, 6)
]

-- Calculate the average score
def average_score : ℚ :=
  let total_score : ℕ := (score_distribution.map (λ (score, count) => score * count)).sum
  (total_score : ℚ) / total_students

-- Theorem to prove
theorem average_score_is_correct :
  average_score = 8380 / 120 := by sorry

end average_score_is_correct_l505_50531


namespace function_range_l505_50588

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem function_range :
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Icc (-1) 2, f x = y ∧
  ∀ x ∈ Set.Icc (-1) 2, f x ∈ Set.Icc 2 6 :=
by sorry

end function_range_l505_50588


namespace product_xyz_l505_50503

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 2/y = 2) 
  (h2 : y + 2/z = 2) 
  (h3 : y ≠ 0) 
  (h4 : z ≠ 0) : x * y * z = 2 := by
  sorry

end product_xyz_l505_50503


namespace biancas_birthday_money_l505_50569

theorem biancas_birthday_money (amount_per_friend : ℕ) (total_amount : ℕ) : 
  amount_per_friend = 6 → total_amount = 30 → total_amount / amount_per_friend = 5 := by
  sorry

end biancas_birthday_money_l505_50569


namespace perpendicular_tangents_ratio_l505_50537

/-- Given a line ax - by - 2 = 0 and a curve y = x^3 intersecting at point P(1, 1),
    if the tangent lines at P are perpendicular, then a/b = -1/3 -/
theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∀ x y, a * x - b * y - 2 = 0 → y = x^3) →  -- Line and curve equations
  (a * 1 - b * 1 - 2 = 0) →                   -- Point P(1, 1) satisfies line equation
  (1 = 1^3) →                                 -- Point P(1, 1) satisfies curve equation
  (∃ k₁ k₂ : ℝ, k₁ * k₂ = -1 ∧                -- Perpendicular tangent lines condition
              k₁ = a / b ∧                    -- Slope of line
              k₂ = 3 * 1^2) →                 -- Slope of curve at P(1, 1)
  a / b = -1 / 3 := by
sorry

end perpendicular_tangents_ratio_l505_50537


namespace isosceles_triangle_perimeter_l505_50541

/-- An isosceles triangle with side lengths that are roots of x^2 - 4x + 3 = 0 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_root : base^2 - 4*base + 3 = 0 ∧ leg^2 - 4*leg + 3 = 0
  is_isosceles : base ≠ leg
  triangle_inequality : base < 2*leg

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2*t.leg

/-- Theorem: The perimeter of the isosceles triangle is 7 -/
theorem isosceles_triangle_perimeter : 
  ∀ t : IsoscelesTriangle, perimeter t = 7 := by
  sorry

end isosceles_triangle_perimeter_l505_50541


namespace cherry_weekly_earnings_l505_50512

/-- Represents the delivery rates for different weight ranges -/
structure DeliveryRates :=
  (kg3to5 : ℝ)
  (kg6to8 : ℝ)
  (kg9to12 : ℝ)
  (kg13to15 : ℝ)

/-- Represents the daily deliveries -/
structure DailyDeliveries :=
  (kg5 : ℕ)
  (kg8 : ℕ)
  (kg10 : ℕ)
  (kg14 : ℕ)

def weekdayRates : DeliveryRates :=
  { kg3to5 := 2.5, kg6to8 := 4, kg9to12 := 6, kg13to15 := 8 }

def weekendRates : DeliveryRates :=
  { kg3to5 := 3, kg6to8 := 5, kg9to12 := 7.5, kg13to15 := 10 }

def weekdayDeliveries : DailyDeliveries :=
  { kg5 := 4, kg8 := 2, kg10 := 3, kg14 := 1 }

def weekendDeliveries : DailyDeliveries :=
  { kg5 := 2, kg8 := 3, kg10 := 0, kg14 := 2 }

def weekdaysInWeek : ℕ := 5
def weekendDaysInWeek : ℕ := 2

/-- Calculates the daily earnings based on rates and deliveries -/
def dailyEarnings (rates : DeliveryRates) (deliveries : DailyDeliveries) : ℝ :=
  rates.kg3to5 * deliveries.kg5 +
  rates.kg6to8 * deliveries.kg8 +
  rates.kg9to12 * deliveries.kg10 +
  rates.kg13to15 * deliveries.kg14

/-- Calculates the total weekly earnings -/
def weeklyEarnings : ℝ :=
  weekdaysInWeek * dailyEarnings weekdayRates weekdayDeliveries +
  weekendDaysInWeek * dailyEarnings weekendRates weekendDeliveries

theorem cherry_weekly_earnings :
  weeklyEarnings = 302 := by sorry

end cherry_weekly_earnings_l505_50512


namespace square_minus_self_sum_l505_50565

theorem square_minus_self_sum : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 := by
  sorry

end square_minus_self_sum_l505_50565


namespace total_ladybugs_count_l505_50535

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem total_ladybugs_count : total_ladybugs = 67082 := by
  sorry

end total_ladybugs_count_l505_50535


namespace basketball_handshakes_l505_50579

theorem basketball_handshakes : 
  let team_size : ℕ := 6
  let referee_count : ℕ := 3
  let inter_team_handshakes := team_size * team_size
  let player_referee_handshakes := (2 * team_size) * referee_count
  inter_team_handshakes + player_referee_handshakes = 72 :=
by sorry

end basketball_handshakes_l505_50579


namespace intersection_equals_open_closed_interval_l505_50576

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_equals_open_closed_interval : M ∩ N = Set.Ioc 1 2 := by
  sorry

end intersection_equals_open_closed_interval_l505_50576


namespace manuscript_cost_calculation_l505_50513

/-- Calculates the total cost of typing a manuscript given the page counts and rates. -/
def manuscript_typing_cost (
  total_pages : ℕ) 
  (revised_once : ℕ) 
  (revised_twice : ℕ) 
  (revised_twice_sets : ℕ) 
  (revised_thrice : ℕ) 
  (revised_thrice_sets : ℕ) 
  (initial_rate : ℕ) 
  (revision_rate : ℕ) 
  (set_rate_thrice : ℕ) 
  (set_rate_twice : ℕ) : ℕ :=
  sorry

theorem manuscript_cost_calculation :
  manuscript_typing_cost 
    250  -- total pages
    80   -- pages revised once
    95   -- pages revised twice
    2    -- sets of 20 pages revised twice
    50   -- pages revised thrice
    3    -- sets of 10 pages revised thrice
    5    -- initial typing rate
    3    -- revision rate
    10   -- flat fee for set of 10 pages revised 3+ times
    15   -- flat fee for set of 20 pages revised 2 times
  = 1775 := by sorry

end manuscript_cost_calculation_l505_50513


namespace root_sum_fraction_equality_l505_50593

theorem root_sum_fraction_equality (r s t : ℝ) : 
  r^3 - 6*r^2 + 11*r - 6 = 0 → 
  s^3 - 6*s^2 + 11*s - 6 = 0 → 
  t^3 - 6*t^2 + 11*t - 6 = 0 → 
  (r+s)/t + (s+t)/r + (t+r)/s = 25/3 :=
by
  sorry

end root_sum_fraction_equality_l505_50593


namespace TUVW_product_l505_50586

def letter_value (c : Char) : ℕ :=
  c.toNat - 'A'.toNat + 1

theorem TUVW_product : 
  (letter_value 'T') * (letter_value 'U') * (letter_value 'V') * (letter_value 'W') = 
  2^3 * 3 * 5 * 7 * 11 * 23 := by
  sorry

end TUVW_product_l505_50586


namespace investment_ratio_equals_return_ratio_l505_50589

/-- Given three investors with investments in some ratio, prove that their investment ratio
    is the same as their return ratio under certain conditions. -/
theorem investment_ratio_equals_return_ratio
  (a b c : ℕ) -- investments of A, B, and C
  (ra rb rc : ℕ) -- returns of A, B, and C
  (h1 : ra = 6 * k ∧ rb = 5 * k ∧ rc = 4 * k) -- return ratio condition
  (h2 : rb = ra + 250) -- B earns 250 more than A
  (h3 : ra + rb + rc = 7250) -- total earnings
  : ∃ (m : ℕ), a = 6 * m ∧ b = 5 * m ∧ c = 4 * m := by
  sorry


end investment_ratio_equals_return_ratio_l505_50589


namespace endomorphism_characterization_l505_50500

/-- An endomorphism of ℤ² --/
def Endomorphism : Type := ℤ × ℤ → ℤ × ℤ

/-- The group operation on ℤ² --/
def add : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a b : ℤ × ℤ) => (a.1 + b.1, a.2 + b.2)

/-- A homomorphism respects the group operation --/
def is_homomorphism (φ : Endomorphism) : Prop :=
  ∀ a b : ℤ × ℤ, φ (add a b) = add (φ a) (φ b)

/-- Linear representation of an endomorphism --/
def linear_form (u v : ℤ × ℤ) : Endomorphism :=
  λ (x : ℤ × ℤ) => (x.1 * u.1 + x.2 * v.1, x.1 * u.2 + x.2 * v.2)

/-- Main theorem: Characterization of endomorphisms of ℤ² --/
theorem endomorphism_characterization :
  ∀ φ : Endomorphism, 
    is_homomorphism φ ↔ ∃ u v : ℤ × ℤ, φ = linear_form u v :=
by sorry

end endomorphism_characterization_l505_50500


namespace daves_trays_l505_50510

/-- Given that Dave can carry 9 trays at a time, picked up 17 trays from one table,
    and made 8 trips in total, prove that he picked up 55 trays from the second table. -/
theorem daves_trays (trays_per_trip : ℕ) (trips : ℕ) (trays_first_table : ℕ)
    (h1 : trays_per_trip = 9)
    (h2 : trips = 8)
    (h3 : trays_first_table = 17) :
    trips * trays_per_trip - trays_first_table = 55 := by
  sorry

end daves_trays_l505_50510


namespace arithmetic_sequence_sum_l505_50546

theorem arithmetic_sequence_sum (a₁ d n : ℕ) (h : n > 0) : 
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  a₁ = 4 ∧ d = 5 ∧ n = 12 → S = 378 := by
  sorry

end arithmetic_sequence_sum_l505_50546


namespace quadratic_coefficient_l505_50556

/-- Given two points on a quadratic function, prove the value of b -/
theorem quadratic_coefficient (a c y₁ y₂ : ℝ) :
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -12 →
  b = -3 := by
sorry

end quadratic_coefficient_l505_50556


namespace hockey_league_games_l505_50536

/-- The total number of games played in a hockey league season -/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264 -/
theorem hockey_league_games :
  total_games 12 4 = 264 := by
  sorry

end hockey_league_games_l505_50536


namespace second_girl_speed_l505_50517

/-- Given two girls walking in opposite directions, prove that the second girl's speed is 3 km/hr -/
theorem second_girl_speed (girl1_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  girl1_speed = 7 ∧ time = 12 ∧ distance = 120 →
  ∃ girl2_speed : ℝ, girl2_speed = 3 ∧ distance = (girl1_speed + girl2_speed) * time :=
by
  sorry

end second_girl_speed_l505_50517


namespace range_of_2a_minus_b_l505_50502

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2*a^2 - a*b - b^2 - 4 = 0) :
  ∃ (k : ℝ), k ≥ 8/3 ∧ 2*a - b = k :=
sorry

end range_of_2a_minus_b_l505_50502


namespace clay_molding_minimum_operations_l505_50528

/-- Represents a clay molding operation -/
structure ClayOperation where
  groups : List (List Nat)
  deriving Repr

/-- The result of applying a clay molding operation -/
def applyOperation (pieces : List Nat) (op : ClayOperation) : List Nat :=
  sorry

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List Nat) : Prop :=
  sorry

/-- The main theorem stating that 2 operations are sufficient and minimal -/
theorem clay_molding_minimum_operations :
  ∃ (op1 op2 : ClayOperation),
    let initial_pieces := List.replicate 111 1
    let after_op1 := applyOperation initial_pieces op1
    let final_pieces := applyOperation after_op1 op2
    (final_pieces.length = 11) ∧
    (allDistinct final_pieces) ∧
    (∀ (op1' op2' : ClayOperation),
      let after_op1' := applyOperation initial_pieces op1'
      let final_pieces' := applyOperation after_op1' op2'
      (final_pieces'.length = 11 ∧ allDistinct final_pieces') →
      ¬∃ (single_op : ClayOperation),
        let result := applyOperation initial_pieces single_op
        (result.length = 11 ∧ allDistinct result)) :=
  sorry

end clay_molding_minimum_operations_l505_50528


namespace unique_pair_for_squared_difference_l505_50590

theorem unique_pair_for_squared_difference : 
  ∃! (a b : ℕ), a^2 - b^2 = 25 ∧ a = 13 ∧ b = 12 :=
by sorry

end unique_pair_for_squared_difference_l505_50590


namespace product_units_digit_base8_l505_50540

theorem product_units_digit_base8 : ∃ (n : ℕ), 
  (505 * 71) % 8 = n ∧ n = ((505 % 8) * (71 % 8)) % 8 := by
  sorry

end product_units_digit_base8_l505_50540
