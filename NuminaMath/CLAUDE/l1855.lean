import Mathlib

namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l1855_185576

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1
theorem theorem_1 (m n : Line) (α : Plane) :
  perpendicular m α → parallel n α → perpendicular_lines m n :=
sorry

-- Theorem 2
theorem theorem_2 (m : Line) (α β γ : Plane) :
  perpendicular_planes α γ → perpendicular_planes β γ → intersect α β m → perpendicular m γ :=
sorry

-- Assumptions
axiom different_lines (m n : Line) : m ≠ n
axiom different_planes (α β γ : Plane) : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l1855_185576


namespace NUMINAMATH_CALUDE_road_length_difference_l1855_185531

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- Conversion factor from meters to kilometers -/
def meters_to_km : ℝ := 1000

theorem road_length_difference :
  telegraph_road_length - (pardee_road_length / meters_to_km) = 150 := by
  sorry

end NUMINAMATH_CALUDE_road_length_difference_l1855_185531


namespace NUMINAMATH_CALUDE_at_least_one_correct_l1855_185582

theorem at_least_one_correct (p_a p_b : ℚ) 
  (h_a : p_a = 3/5) 
  (h_b : p_b = 2/5) : 
  1 - (1 - p_a) * (1 - p_b) = 19/25 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_correct_l1855_185582


namespace NUMINAMATH_CALUDE_number_of_possible_orders_l1855_185574

/-- The number of documents --/
def n : ℕ := 10

/-- The number of documents before the confirmed reviewed document --/
def m : ℕ := 8

/-- Calculates the number of possible orders for the remaining documents --/
def possibleOrders : ℕ := 
  Finset.sum (Finset.range (m + 1)) (fun k => (Nat.choose m k) * (k + 2))

/-- Theorem stating the number of possible orders --/
theorem number_of_possible_orders : possibleOrders = 1440 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_orders_l1855_185574


namespace NUMINAMATH_CALUDE_pencil_count_l1855_185581

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := 71 - 30

/-- The number of pencils Mike added to the drawer -/
def added_pencils : ℕ := 30

/-- The total number of pencils after Mike's addition -/
def total_pencils : ℕ := 71

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by
  sorry

#eval original_pencils -- This will output 41

end NUMINAMATH_CALUDE_pencil_count_l1855_185581


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l1855_185542

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (Real.exp x)

theorem tangent_line_at_zero (x y : ℝ) :
  (∃ (m : ℝ), HasDerivAt f m 0 ∧ m = -1) →
  f 0 = 1 →
  (x + y - 1 = 0 ↔ y - f 0 = m * (x - 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l1855_185542


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_l1855_185538

-- Define a type for planes
structure Plane where
  -- We don't need to specify the exact properties of a plane for this problem

-- Define a perpendicular relation between planes
def perpendicular (p q : Plane) : Prop := sorry

-- Define a parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- Define an intersecting relation between planes
def intersecting (p q : Plane) : Prop := sorry

-- State the theorem
theorem planes_perpendicular_to_same_plane 
  (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : α ≠ γ) (h3 : β ≠ γ) 
  (h4 : perpendicular α γ) (h5 : perpendicular β γ) : 
  parallel α β ∨ intersecting α β := by
  sorry


end NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_l1855_185538


namespace NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l1855_185586

theorem proposition_p_sufficient_not_necessary_for_q (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 1| ≥ m →
   ∃ x₀ : ℝ, x₀^2 - 2*m*x₀ + m^2 + m - 3 = 0) ∧
  (∃ m : ℝ, (∃ x₀ : ℝ, x₀^2 - 2*m*x₀ + m^2 + m - 3 = 0) ∧
   ¬(∀ x : ℝ, |x + 1| + |x - 1| ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l1855_185586


namespace NUMINAMATH_CALUDE_penny_species_count_l1855_185532

/-- The number of distinct species Penny identified at the aquarium -/
def distinctSpecies (sharks eels whales dolphins rays octopuses uniqueSpecies doubleCounted : ℕ) : ℕ :=
  sharks + eels + whales + dolphins + rays + octopuses - doubleCounted

/-- Theorem stating the number of distinct species Penny identified -/
theorem penny_species_count :
  distinctSpecies 35 15 5 12 8 25 6 3 = 97 := by
  sorry

end NUMINAMATH_CALUDE_penny_species_count_l1855_185532


namespace NUMINAMATH_CALUDE_correct_fraction_l1855_185528

/-- The number of quarters Roger has -/
def total_quarters : ℕ := 22

/-- The number of states that joined the union during 1800-1809 -/
def states_1800_1809 : ℕ := 5

/-- The fraction of quarters representing states that joined during 1800-1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_quarters

theorem correct_fraction :
  fraction_1800_1809 = 5 / 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_fraction_l1855_185528


namespace NUMINAMATH_CALUDE_larger_number_proof_l1855_185510

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (∃ (k : ℕ+), Nat.lcm a b = 23 * 15 * 16 * k) →
  (max a b = 368) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1855_185510


namespace NUMINAMATH_CALUDE_b_share_is_600_l1855_185524

/-- Given a partnership where A invests 3 times as much as B, and B invests two-thirds of what C invests,
    this function calculates B's share of the profit when the total profit is 3300 Rs. -/
def calculate_B_share (total_profit : ℚ) : ℚ :=
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 2/3
  let c_ratio : ℚ := 1
  let total_ratio : ℚ := a_ratio + b_ratio + c_ratio
  (b_ratio / total_ratio) * total_profit

/-- Theorem stating that B's share of the profit is 600 Rs -/
theorem b_share_is_600 :
  calculate_B_share 3300 = 600 := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_600_l1855_185524


namespace NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1855_185570

theorem pythagorean_triple_6_8_10 : 
  ∃ (a b c : ℕ+), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 := by
  sorry

#check pythagorean_triple_6_8_10

end NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1855_185570


namespace NUMINAMATH_CALUDE_length_breadth_difference_l1855_185513

/-- Represents a rectangular plot with given dimensions and fencing cost. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Theorem stating the difference between length and breadth of the plot. -/
theorem length_breadth_difference (plot : RectangularPlot)
  (h1 : plot.length = 75)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.total_fencing_cost = (2 * plot.length + 2 * plot.breadth) * plot.fencing_cost_per_meter) :
  plot.length - plot.breadth = 50 := by
  sorry

#check length_breadth_difference

end NUMINAMATH_CALUDE_length_breadth_difference_l1855_185513


namespace NUMINAMATH_CALUDE_power_sum_modulo_l1855_185537

theorem power_sum_modulo (n : ℕ) :
  (Nat.pow 7 2008 + Nat.pow 9 2008) % 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_modulo_l1855_185537


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l1855_185578

/-- Given two curves y = (1/2e)x^2 and y = a ln x with a common tangent at their common point P(s,t), prove that a = 1 --/
theorem common_tangent_implies_a_equals_one (e : ℝ) (a s t : ℝ) : 
  (t = (1/(2*Real.exp 1))*s^2) → 
  (t = a * Real.log s) → 
  ((s / Real.exp 1) = (a / s)) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l1855_185578


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1855_185569

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l1855_185569


namespace NUMINAMATH_CALUDE_inequality_proof_l1855_185555

theorem inequality_proof (w x y z : ℝ) (h : w^2 + y^2 ≤ 1) :
  (w*x + y*z - 1)^2 ≥ (w^2 + y^2 - 1)*(x^2 + z^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1855_185555


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1855_185595

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define shapes
inductive Shape
  | Square
  | Circle

-- Define colors
inductive Color
  | Black
  | White

-- Define a coloring function
def ColoringFunction := Point → Color

-- Define similarity between sets of points
def SimilarSets (s1 s2 : Set Point) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ ∀ (p1 p2 : Point), p1 ∈ s1 → p2 ∈ s2 →
    ∃ (q1 q2 : Point), q1 ∈ s2 ∧ q2 ∈ s2 ∧
      (q1.x - q2.x)^2 + (q1.y - q2.y)^2 = k * ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem exists_valid_coloring :
  ∃ (f : Shape → ColoringFunction),
    (∀ (s : Shape) (p : Point), f s p = Color.Black ∨ f s p = Color.White) ∧
    SimilarSets {p | f Shape.Square p = Color.White} {p | f Shape.Circle p = Color.White} ∧
    SimilarSets {p | f Shape.Square p = Color.Black} {p | f Shape.Circle p = Color.Black} :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1855_185595


namespace NUMINAMATH_CALUDE_percentage_of_150_l1855_185540

theorem percentage_of_150 : (1 / 5 : ℚ) / 100 * 150 = 0.3 := by sorry

end NUMINAMATH_CALUDE_percentage_of_150_l1855_185540


namespace NUMINAMATH_CALUDE_village_language_problem_l1855_185541

theorem village_language_problem (total_population : ℕ) 
  (tamil_speakers : ℕ) (english_speakers : ℕ) (hindi_probability : ℚ) :
  total_population = 1024 →
  tamil_speakers = 720 →
  english_speakers = 562 →
  hindi_probability = 0.0859375 →
  ∃ (both_speakers : ℕ),
    both_speakers = 434 ∧
    total_population = tamil_speakers + english_speakers - both_speakers + 
      (↑total_population * hindi_probability).floor := by
  sorry

end NUMINAMATH_CALUDE_village_language_problem_l1855_185541


namespace NUMINAMATH_CALUDE_apple_box_weight_l1855_185516

/-- Given a box of apples with total weight and weight after removing half the apples,
    prove the weight of the box and the weight of the apples. -/
theorem apple_box_weight (total_weight : ℝ) (half_removed_weight : ℝ)
    (h1 : total_weight = 62.8)
    (h2 : half_removed_weight = 31.8) :
    ∃ (box_weight apple_weight : ℝ),
      box_weight = 0.8 ∧
      apple_weight = 62 ∧
      total_weight = box_weight + apple_weight ∧
      half_removed_weight = box_weight + apple_weight / 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_box_weight_l1855_185516


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1855_185588

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1855_185588


namespace NUMINAMATH_CALUDE_unique_zero_iff_a_nonpositive_l1855_185597

/-- A function f(x) = x^3 - 3ax has a unique zero if and only if a ≤ 0 -/
theorem unique_zero_iff_a_nonpositive (a : ℝ) :
  (∃! x, x^3 - 3*a*x = 0) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_unique_zero_iff_a_nonpositive_l1855_185597


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1855_185571

theorem complex_equation_solution :
  let z : ℂ := ((1 - Complex.I)^2 + 3 * (1 + Complex.I)) / (2 - Complex.I)
  ∃ (a b : ℝ), z^2 + a*z + b = 1 - Complex.I ∧ z = 1 + Complex.I ∧ a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1855_185571


namespace NUMINAMATH_CALUDE_steven_owes_jeremy_l1855_185559

/-- The amount Steven owes Jeremy for cleaning rooms -/
theorem steven_owes_jeremy (rate : ℚ) (rooms : ℚ) : rate = 13/3 → rooms = 5/2 → rate * rooms = 65/6 := by
  sorry

end NUMINAMATH_CALUDE_steven_owes_jeremy_l1855_185559


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1855_185592

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t 
    where g(-1) = 4, prove that 16p - 8q + 4r - 2s + t = 64 -/
theorem polynomial_value_theorem 
  (p q r s t : ℝ) 
  (g : ℝ → ℝ)
  (h1 : ∀ x, g x = p * x^4 + q * x^3 + r * x^2 + s * x + t)
  (h2 : g (-1) = 4) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1855_185592


namespace NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_l1855_185563

-- Define the angles
variable (A B C D F G : ℝ)

-- Define the condition that these angles form a quadrilateral
variable (h : IsQuadrilateral A B C D F G)

-- State the theorem
theorem sum_of_angles_in_quadrilateral :
  A + B + C + D + F + G = 360 :=
sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_quadrilateral_l1855_185563


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1855_185545

theorem rectangle_dimension_change (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_L := L * (1 - 0.25)
  let new_W := W * (1 + 1/3)
  new_L * new_W = L * W := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1855_185545


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l1855_185589

theorem intersection_point_satisfies_equations :
  let x : ℚ := 75 / 8
  let y : ℚ := 15 / 8
  (3 * x^2 - 12 * y^2 = 48) ∧ (y = -1/3 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l1855_185589


namespace NUMINAMATH_CALUDE_area_midpoint_rectangle_l1855_185508

/-- Given a rectangle EFGH with width w and height h, and points P and Q that are
    midpoints of the longer sides EF and GH respectively, the area of EPGQ is
    half the area of EFGH. -/
theorem area_midpoint_rectangle (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let rect_area := w * h
  let midpoint_rect_area := (w / 2) * h
  midpoint_rect_area = rect_area / 2 := by
  sorry

#check area_midpoint_rectangle

end NUMINAMATH_CALUDE_area_midpoint_rectangle_l1855_185508


namespace NUMINAMATH_CALUDE_recess_time_calculation_l1855_185551

/-- Calculates the total recess time based on grade distribution -/
def total_recess_time (normal_recess : ℕ) 
  (extra_time_A extra_time_B extra_time_C extra_time_D extra_time_E extra_time_F : ℤ)
  (num_A num_B num_C num_D num_E num_F : ℕ) : ℤ :=
  normal_recess + 
  extra_time_A * num_A + 
  extra_time_B * num_B + 
  extra_time_C * num_C + 
  extra_time_D * num_D + 
  extra_time_E * num_E + 
  extra_time_F * num_F

theorem recess_time_calculation :
  total_recess_time 20 4 3 2 1 (-1) (-2) 10 12 14 5 3 2 = 122 := by
  sorry

end NUMINAMATH_CALUDE_recess_time_calculation_l1855_185551


namespace NUMINAMATH_CALUDE_inductive_reasoning_is_specific_to_general_l1855_185511

-- Define the types of reasoning
inductive ReasoningType
  | Analogical
  | Deductive
  | Inductive
  | Emotional

-- Define the direction of reasoning
inductive ReasoningDirection
  | SpecificToGeneral
  | GeneralToSpecific
  | Other

-- Function to get the direction of a reasoning type
def getReasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.SpecificToGeneral
  | _ => ReasoningDirection.Other

-- Theorem statement
theorem inductive_reasoning_is_specific_to_general :
  ∃ (rt : ReasoningType), getReasoningDirection rt = ReasoningDirection.SpecificToGeneral ∧
  rt = ReasoningType.Inductive :=
sorry

end NUMINAMATH_CALUDE_inductive_reasoning_is_specific_to_general_l1855_185511


namespace NUMINAMATH_CALUDE_kitten_growth_l1855_185579

/-- The length of a kitten after doubling twice from an initial length of 4 inches. -/
def kitten_length : ℕ := 16

/-- The initial length of the kitten in inches. -/
def initial_length : ℕ := 4

/-- Doubling function -/
def double (n : ℕ) : ℕ := 2 * n

theorem kitten_growth : kitten_length = double (double initial_length) := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_l1855_185579


namespace NUMINAMATH_CALUDE_smallest_w_l1855_185587

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : w > 0 →
  is_factor (2^7) (2880 * w) →
  is_factor (3^4) (2880 * w) →
  is_factor (5^3) (2880 * w) →
  is_factor (7^3) (2880 * w) →
  is_factor (11^2) (2880 * w) →
  w ≥ 37348700 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_w_l1855_185587


namespace NUMINAMATH_CALUDE_seokjin_math_score_l1855_185549

/-- Given Seokjin's scores and average, prove his math score -/
theorem seokjin_math_score 
  (korean_score : ℕ) 
  (english_score : ℕ) 
  (average_score : ℕ) 
  (h1 : korean_score = 93)
  (h2 : english_score = 91)
  (h3 : average_score = 89)
  (h4 : (korean_score + english_score + math_score) / 3 = average_score) :
  math_score = 83 :=
by
  sorry

end NUMINAMATH_CALUDE_seokjin_math_score_l1855_185549


namespace NUMINAMATH_CALUDE_sum_of_ab_is_fifteen_l1855_185548

-- Define the set of digits
def Digit := Fin 10

-- Define the property of being four different digits
def FourDifferentDigits (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Define the property of (A+B)/(C+D) being an integer
def SumRatioIsInteger (a b c d : Digit) : Prop :=
  ∃ (k : ℕ), (a.val + b.val : ℕ) = k * (c.val + d.val)

-- Define the property of C and D being non-zero
def NonZeroCD (c d : Digit) : Prop :=
  c.val ≠ 0 ∧ d.val ≠ 0

-- Define the property of C and D being as small as possible
def MinimalCD (c d : Digit) : Prop :=
  ∀ (c' d' : Digit), NonZeroCD c' d' → c.val + d.val ≤ c'.val + d'.val

theorem sum_of_ab_is_fifteen :
  ∀ (a b c d : Digit),
    FourDifferentDigits a b c d →
    SumRatioIsInteger a b c d →
    NonZeroCD c d →
    MinimalCD c d →
    a.val + b.val = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ab_is_fifteen_l1855_185548


namespace NUMINAMATH_CALUDE_box_volume_percentage_l1855_185577

/-- The percentage of volume occupied by 4-inch cubes in a rectangular box -/
theorem box_volume_percentage :
  let box_length : ℕ := 8
  let box_width : ℕ := 6
  let box_height : ℕ := 12
  let cube_size : ℕ := 4
  let cubes_length : ℕ := box_length / cube_size
  let cubes_width : ℕ := box_width / cube_size
  let cubes_height : ℕ := box_height / cube_size
  let total_cubes : ℕ := cubes_length * cubes_width * cubes_height
  let cubes_volume : ℕ := total_cubes * (cube_size ^ 3)
  let box_volume : ℕ := box_length * box_width * box_height
  (cubes_volume : ℚ) / (box_volume : ℚ) = 2 / 3 :=
by
  sorry

#check box_volume_percentage

end NUMINAMATH_CALUDE_box_volume_percentage_l1855_185577


namespace NUMINAMATH_CALUDE_blue_area_ratio_l1855_185501

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  /-- Side length of the flag -/
  side : ℝ
  /-- Width of the cross -/
  cross_width : ℝ
  /-- Assumption that the cross (including blue center) is 36% of total area -/
  cross_area_ratio : cross_width * (4 * side - cross_width) / (side * side) = 0.36

/-- Theorem stating that the blue area is 2% of the total flag area -/
theorem blue_area_ratio (flag : SquareFlag) : 
  (flag.cross_width / flag.side) ^ 2 = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_blue_area_ratio_l1855_185501


namespace NUMINAMATH_CALUDE_radius_is_ten_l1855_185525

/-- A square with a circle tangent to two adjacent sides -/
structure TangentSquare where
  /-- Side length of the square -/
  side : ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- Length of segment cut off from vertices B and D -/
  tangent_segment : ℝ
  /-- Length of segment cut off from one non-tangent side -/
  intersect_segment1 : ℝ
  /-- Length of segment cut off from the other non-tangent side -/
  intersect_segment2 : ℝ
  /-- The circle is tangent to two adjacent sides -/
  tangent_condition : side = radius + tangent_segment
  /-- The circle intersects the other two sides -/
  intersect_condition : side = radius + intersect_segment1 + intersect_segment2

/-- The radius of the circle is 10 given the specific measurements -/
theorem radius_is_ten (ts : TangentSquare) 
  (h1 : ts.tangent_segment = 8)
  (h2 : ts.intersect_segment1 = 4)
  (h3 : ts.intersect_segment2 = 2) : 
  ts.radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_ten_l1855_185525


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1855_185580

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6*m*x + 6

-- Define what it means for f to be decreasing on the interval (-∞, 3]
def is_decreasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 3 → f m x > f m y

-- State the theorem
theorem sufficient_not_necessary_condition :
  (m = 1 → is_decreasing_on_interval m) ∧
  ¬(is_decreasing_on_interval m → m = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1855_185580


namespace NUMINAMATH_CALUDE_ocean_depth_l1855_185562

/-- The depth of the ocean given echo sounder measurements -/
theorem ocean_depth (t : ℝ) (v : ℝ) (h : ℝ) : t = 5 → v = 1.5 → h = (t * v * 1000) / 2 → h = 3750 :=
by sorry

end NUMINAMATH_CALUDE_ocean_depth_l1855_185562


namespace NUMINAMATH_CALUDE_gift_exchange_probability_l1855_185526

theorem gift_exchange_probability :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 4
  let total_people : ℕ := num_boys + num_girls
  let total_configurations : ℕ := num_boys ^ total_people
  let valid_configurations : ℕ := 288

  (valid_configurations : ℚ) / total_configurations = 9 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_gift_exchange_probability_l1855_185526


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equals_405_l1855_185558

theorem arithmetic_sequence_equals_405 : ((306 / 34) * 15) + 270 = 405 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equals_405_l1855_185558


namespace NUMINAMATH_CALUDE_joshua_needs_32_cents_l1855_185504

/-- The amount of additional cents Joshua needs to purchase a pen -/
def additional_cents_needed (pen_cost : ℕ) (joshua_money : ℕ) (borrowed_amount : ℕ) : ℕ :=
  pen_cost - (joshua_money + borrowed_amount)

/-- Theorem: Joshua needs 32 more cents to buy the pen -/
theorem joshua_needs_32_cents :
  additional_cents_needed 600 500 68 = 32 := by
  sorry

end NUMINAMATH_CALUDE_joshua_needs_32_cents_l1855_185504


namespace NUMINAMATH_CALUDE_dans_initial_money_l1855_185565

/-- Dan's initial amount of money, given his remaining money and the cost of a candy bar. -/
def initial_money (remaining : ℕ) (candy_cost : ℕ) : ℕ :=
  remaining + candy_cost

theorem dans_initial_money :
  initial_money 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l1855_185565


namespace NUMINAMATH_CALUDE_factorization_equality_l1855_185596

theorem factorization_equality (a b c : ℝ) :
  -14 * a * b * c - 7 * a * b + 49 * a * b^2 * c = -7 * a * b * (2 * c + 1 - 7 * b * c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1855_185596


namespace NUMINAMATH_CALUDE_no_valid_n_exists_l1855_185521

theorem no_valid_n_exists : ¬ ∃ (n : ℕ), 0 < n ∧ n < 200 ∧ 
  ∃ (m : ℕ), 4 ∣ m ∧ ∃ (k : ℕ), m = k^2 ∧
  ∃ (r : ℕ), (r^2 - n*r + m = 0) ∧ ((r+1)^2 - n*(r+1) + m = 0) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_exists_l1855_185521


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l1855_185575

theorem opposite_of_negative_three : -((-3 : ℤ)) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l1855_185575


namespace NUMINAMATH_CALUDE_area_ratio_quadrilateral_to_dodecagon_l1855_185517

/-- Regular dodecagon with vertices ABCDEFGHIJKL -/
structure RegularDodecagon where
  vertices : Fin 12 → ℝ × ℝ
  is_regular : sorry

/-- Area of a regular dodecagon -/
def area_dodecagon (d : RegularDodecagon) : ℝ := sorry

/-- Area of quadrilateral ACEG in a regular dodecagon -/
def area_quadrilateral_ACEG (d : RegularDodecagon) : ℝ := sorry

/-- Theorem: The ratio of the area of quadrilateral ACEG to the area of a regular dodecagon is 1/(3√3) -/
theorem area_ratio_quadrilateral_to_dodecagon (d : RegularDodecagon) :
  area_quadrilateral_ACEG d / area_dodecagon d = 1 / (3 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_area_ratio_quadrilateral_to_dodecagon_l1855_185517


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1855_185509

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  (X^4 : Polynomial ℝ) + 3 * X^2 - 2 = (X^2 - 4 * X + 3) * q + r ∧ 
  r = 88 * X - 59 ∧ 
  r.degree < (X^2 - 4 * X + 3).degree := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1855_185509


namespace NUMINAMATH_CALUDE_fuel_price_per_gallon_l1855_185518

/-- Given the following conditions:
    1. The total amount of fuel is 100 gallons
    2. The fuel consumption rate is $0.40 worth of fuel per hour
    3. It takes 175 hours to consume all the fuel
    Prove that the price per gallon of fuel is $0.70 -/
theorem fuel_price_per_gallon 
  (total_fuel : ℝ) 
  (consumption_rate : ℝ) 
  (total_hours : ℝ) 
  (h1 : total_fuel = 100) 
  (h2 : consumption_rate = 0.40) 
  (h3 : total_hours = 175) : 
  (consumption_rate * total_hours) / total_fuel = 0.70 := by
  sorry

#check fuel_price_per_gallon

end NUMINAMATH_CALUDE_fuel_price_per_gallon_l1855_185518


namespace NUMINAMATH_CALUDE_sqrt_of_square_positive_l1855_185585

theorem sqrt_of_square_positive (a : ℝ) (h : a > 0) : Real.sqrt (a^2) = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_positive_l1855_185585


namespace NUMINAMATH_CALUDE_seven_people_circular_arrangement_l1855_185583

/-- The number of ways to arrange n people around a circular table -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n people around a circular table,
    where k specific people must sit together -/
def circularArrangementsWithGroup (n k : ℕ) : ℕ :=
  circularArrangements (n - k + 1) * (k - 1).factorial

theorem seven_people_circular_arrangement :
  circularArrangementsWithGroup 7 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_circular_arrangement_l1855_185583


namespace NUMINAMATH_CALUDE_five_integers_average_l1855_185591

theorem five_integers_average (a b c d e : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a + b + c + d + e : ℚ) / 5 = 7 ∧
  ∀ (x y z w v : ℕ+), 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
    y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
    z ≠ w ∧ z ≠ v ∧
    w ≠ v ∧
    (x + y + z + w + v : ℚ) / 5 = 7 →
    (max a b - min a b : ℤ) ≥ (max x y - min x y : ℤ) ∧
    (max a c - min a c : ℤ) ≥ (max x z - min x z : ℤ) ∧
    (max a d - min a d : ℤ) ≥ (max x w - min x w : ℤ) ∧
    (max a e - min a e : ℤ) ≥ (max x v - min x v : ℤ) ∧
    (max b c - min b c : ℤ) ≥ (max y z - min y z : ℤ) ∧
    (max b d - min b d : ℤ) ≥ (max y w - min y w : ℤ) ∧
    (max b e - min b e : ℤ) ≥ (max y v - min y v : ℤ) ∧
    (max c d - min c d : ℤ) ≥ (max z w - min z w : ℤ) ∧
    (max c e - min c e : ℤ) ≥ (max z v - min z v : ℤ) ∧
    (max d e - min d e : ℤ) ≥ (max w v - min w v : ℤ) →
  (b + c + d : ℚ) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_five_integers_average_l1855_185591


namespace NUMINAMATH_CALUDE_cakes_per_person_l1855_185556

theorem cakes_per_person (total_cakes : ℕ) (num_friends : ℕ) 
  (h1 : total_cakes = 32) 
  (h2 : num_friends = 8) 
  (h3 : total_cakes % num_friends = 0) : 
  total_cakes / num_friends = 4 := by
sorry

end NUMINAMATH_CALUDE_cakes_per_person_l1855_185556


namespace NUMINAMATH_CALUDE_large_number_with_specific_divisors_l1855_185505

/-- A function that returns the list of divisors of a natural number -/
def divisors (n : ℕ) : List ℕ := sorry

/-- A predicate that checks if a list of natural numbers has alternating parity -/
def hasAlternatingParity (l : List ℕ) : Prop := sorry

theorem large_number_with_specific_divisors (n : ℕ) 
  (h1 : (divisors n).length = 1000)
  (h2 : hasAlternatingParity (divisors n)) :
  n > 10^150 := by sorry

end NUMINAMATH_CALUDE_large_number_with_specific_divisors_l1855_185505


namespace NUMINAMATH_CALUDE_equal_sum_sequence_properties_l1855_185554

/-- An equal sum sequence is a sequence where each term plus the previous term
    equals the same constant, starting from the second term. -/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k

theorem equal_sum_sequence_properties (a : ℕ → ℝ) (h : EqualSumSequence a) :
  (∀ n : ℕ, n ≥ 1 → a n = a (n + 2)) ∧
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → Odd m ∧ Odd n → a m = a n) ∧
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → Even m ∧ Even n → a m = a n) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_properties_l1855_185554


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l1855_185502

/-- Represents the number of people in each age group -/
structure AgeGroups where
  over40 : ℕ
  between30and40 : ℕ
  under30 : ℕ

/-- Represents the sample sizes for each age group -/
structure SampleSizes where
  over40 : ℕ
  between30and40 : ℕ
  under30 : ℕ

/-- Calculates the stratified sample sizes given the total population, total sample size, and age group sizes -/
def calculateStratifiedSample (totalPopulation : ℕ) (totalSampleSize : ℕ) (ageGroups : AgeGroups) : SampleSizes :=
  let ratio := totalSampleSize / totalPopulation
  { over40 := ageGroups.over40 * ratio,
    between30and40 := ageGroups.between30and40 * ratio,
    under30 := ageGroups.under30 * ratio }

theorem stratified_sample_theorem (totalPopulation : ℕ) (totalSampleSize : ℕ) (ageGroups : AgeGroups) :
  totalPopulation = 300 →
  totalSampleSize = 30 →
  ageGroups.over40 = 50 →
  ageGroups.between30and40 = 150 →
  ageGroups.under30 = 100 →
  let sample := calculateStratifiedSample totalPopulation totalSampleSize ageGroups
  sample.over40 = 5 ∧ sample.between30and40 = 15 ∧ sample.under30 = 10 :=
by sorry


end NUMINAMATH_CALUDE_stratified_sample_theorem_l1855_185502


namespace NUMINAMATH_CALUDE_no_real_solution_quadratic_l1855_185515

theorem no_real_solution_quadratic : ¬ ∃ x : ℝ, x^2 + 3*x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_no_real_solution_quadratic_l1855_185515


namespace NUMINAMATH_CALUDE_angle_between_legs_l1855_185566

/-- Given two equal right triangles ABC and ADC with common hypotenuse AC,
    where the angle between planes ABC and ADC is α,
    and the angle between equal legs AB and AD is β,
    prove that the angle between legs BC and CD is
    2 * arcsin(sqrt(sin((α + β)/2) * sin((α - β)/2))). -/
theorem angle_between_legs (α β : Real) :
  let angle_between_planes := α
  let angle_between_equal_legs := β
  let angle_between_BC_CD := 2 * Real.arcsin (Real.sqrt (Real.sin ((α + β) / 2) * Real.sin ((α - β) / 2)))
  angle_between_BC_CD = 2 * Real.arcsin (Real.sqrt (Real.sin ((α + β) / 2) * Real.sin ((α - β) / 2))) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_legs_l1855_185566


namespace NUMINAMATH_CALUDE_event_probability_comparison_l1855_185530

theorem event_probability_comparison (v : ℝ) (n : ℕ) (h₁ : v = 0.1) (h₂ : n = 998) :
  (n.choose 99 : ℝ) * v^99 * (1 - v)^(n - 99) > (n.choose 100 : ℝ) * v^100 * (1 - v)^(n - 100) :=
sorry

end NUMINAMATH_CALUDE_event_probability_comparison_l1855_185530


namespace NUMINAMATH_CALUDE_soccer_team_matches_l1855_185546

theorem soccer_team_matches :
  ∀ (initial_matches : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_matches / 5) →
    ∀ (total_matches : ℕ),
      total_matches = initial_matches + 12 →
      (initial_wins + 8 : ℚ) / total_matches = 11 / 20 →
      total_matches = 21 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_matches_l1855_185546


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l1855_185506

/-- Given a point M with coordinates (3, -4), its symmetric point with respect to the x-axis has coordinates (3, 4) -/
theorem symmetric_point_wrt_x_axis :
  let M : ℝ × ℝ := (3, -4)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point M = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l1855_185506


namespace NUMINAMATH_CALUDE_distance_covered_l1855_185584

/-- Proves that the distance covered is 30 km given the conditions of the problem -/
theorem distance_covered (D : ℝ) (S : ℝ) : 
  (D / 5 = D / S + 2) →    -- Abhay takes 2 hours more than Sameer
  (D / 10 = D / S - 1) →   -- If Abhay doubles his speed, he takes 1 hour less than Sameer
  D = 30 := by             -- The distance covered is 30 km
sorry

end NUMINAMATH_CALUDE_distance_covered_l1855_185584


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l1855_185534

theorem cos_alpha_plus_pi_fourth (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is an obtuse angle
  (h2 : Real.sin (α - 3*π/4) = 3/5) :
  Real.cos (α + π/4) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l1855_185534


namespace NUMINAMATH_CALUDE_smallest_a_value_exists_polynomial_with_61_l1855_185567

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure Polynomial4 (α : Type) [Ring α] where
  a : α
  b : α
  c : α

/-- Predicate to check if a list of integers are the roots of a polynomial -/
def are_roots (p : Polynomial4 ℤ) (roots : List ℤ) : Prop :=
  roots.length = 4 ∧
  (∀ x ∈ roots, x > 0) ∧
  (∀ x ∈ roots, x^4 - p.a * x^3 + p.b * x^2 - p.c * x + 5160 = 0)

/-- The main theorem statement -/
theorem smallest_a_value (p : Polynomial4 ℤ) (roots : List ℤ) :
  are_roots p roots → p.a ≥ 61 := by sorry

/-- The existence of a polynomial with a = 61 -/
theorem exists_polynomial_with_61 :
  ∃ (p : Polynomial4 ℤ) (roots : List ℤ), are_roots p roots ∧ p.a = 61 := by sorry

end NUMINAMATH_CALUDE_smallest_a_value_exists_polynomial_with_61_l1855_185567


namespace NUMINAMATH_CALUDE_total_visible_area_formula_l1855_185590

/-- The total area of the visible large rectangle and the additional rectangle, excluding the hole -/
def total_visible_area (x : ℝ) : ℝ :=
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3) + (x + 2) * x

/-- Theorem stating that the total visible area is equal to 26x + 36 -/
theorem total_visible_area_formula (x : ℝ) :
  total_visible_area x = 26 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_total_visible_area_formula_l1855_185590


namespace NUMINAMATH_CALUDE_identical_numbers_iff_even_l1855_185503

/-- A function that represents the operation of selecting two numbers and replacing them with their sum. -/
def sumOperation (numbers : List ℕ) : List (List ℕ) :=
  sorry

/-- A predicate that checks if all numbers in a list are identical. -/
def allIdentical (numbers : List ℕ) : Prop :=
  sorry

/-- A proposition stating that it's possible to transform n numbers into n identical numbers
    using the sum operation if and only if n is even. -/
theorem identical_numbers_iff_even (n : ℕ) (h : n ≥ 2) :
  (∃ (initial : List ℕ) (final : List ℕ),
    initial.length = n ∧
    final.length = n ∧
    allIdentical final ∧
    final ∈ sumOperation initial) ↔ Even n :=
  sorry

end NUMINAMATH_CALUDE_identical_numbers_iff_even_l1855_185503


namespace NUMINAMATH_CALUDE_points_collinear_if_linear_combination_l1855_185593

/-- Four points in space are collinear if one is a linear combination of the others -/
theorem points_collinear_if_linear_combination (P A B C : EuclideanSpace ℝ (Fin 3)) :
  (C - P) = (1/4 : ℝ) • (A - P) + (3/4 : ℝ) • (B - P) →
  ∃ (t : ℝ), C - A = t • (B - A) :=
by sorry

end NUMINAMATH_CALUDE_points_collinear_if_linear_combination_l1855_185593


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1855_185520

theorem simplify_complex_fraction (x : ℝ) (h : x ≠ 2) :
  ((x + 1) / (x - 2) - 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = 3 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1855_185520


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1855_185598

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 33 ∧ 
  (∀ (m : ℕ), m < n → ¬(87 ∣ (13605 - m))) ∧ 
  (87 ∣ (13605 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1855_185598


namespace NUMINAMATH_CALUDE_line_circle_intersect_l1855_185564

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates of the form ρsinθ = k -/
structure PolarLine where
  k : ℝ

/-- Represents a circle in polar coordinates of the form ρ = asinθ -/
structure PolarCircle where
  a : ℝ

/-- Check if a point lies on a polar line -/
def pointOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  p.ρ * Real.sin p.θ = l.k

/-- Check if a point lies on a polar circle -/
def pointOnCircle (p : PolarPoint) (c : PolarCircle) : Prop :=
  p.ρ = c.a * Real.sin p.θ

/-- Definition of intersection between a polar line and a polar circle -/
def intersect (l : PolarLine) (c : PolarCircle) : Prop :=
  ∃ p : PolarPoint, pointOnLine p l ∧ pointOnCircle p c

theorem line_circle_intersect (l : PolarLine) (c : PolarCircle) 
    (h1 : l.k = 2) (h2 : c.a = 4) : intersect l c := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersect_l1855_185564


namespace NUMINAMATH_CALUDE_circle_inside_parabola_radius_l1855_185536

/-- A circle inside a parabola y = 4x^2, tangent at two points, has radius a^2/4 -/
theorem circle_inside_parabola_radius (a : ℝ) :
  let parabola := fun x : ℝ => 4 * x^2
  let tangent_point1 := (a, parabola a)
  let tangent_point2 := (-a, parabola (-a))
  let circle_center := (0, a^2)
  let radius := a^2 / 4
  (∀ x y, (x - 0)^2 + (y - a^2)^2 = radius^2 → y ≤ parabola x) ∧
  (circle_center.1 - tangent_point1.1)^2 + (circle_center.2 - tangent_point1.2)^2 = radius^2 ∧
  (circle_center.1 - tangent_point2.1)^2 + (circle_center.2 - tangent_point2.2)^2 = radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_inside_parabola_radius_l1855_185536


namespace NUMINAMATH_CALUDE_omelette_combinations_l1855_185500

/-- The number of available fillings for omelettes -/
def num_fillings : ℕ := 8

/-- The number of egg choices for the omelette base -/
def num_egg_choices : ℕ := 4

/-- The total number of omelette combinations -/
def total_combinations : ℕ := 2^num_fillings * num_egg_choices

theorem omelette_combinations : total_combinations = 1024 := by
  sorry

end NUMINAMATH_CALUDE_omelette_combinations_l1855_185500


namespace NUMINAMATH_CALUDE_special_triangle_area_l1855_185514

-- Define a right triangle with a 30° angle and hypotenuse of 20 inches
def special_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem for right triangle
  c = 20 ∧  -- Hypotenuse length
  a / c = 1 / 2  -- Sine of 30° angle (opposite / hypotenuse)

-- Theorem statement
theorem special_triangle_area (a b c : ℝ) 
  (h : special_triangle a b c) : a * b / 2 = 50 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_l1855_185514


namespace NUMINAMATH_CALUDE_pyramid_theorem_l1855_185519

/-- Represents a row in the pyramid -/
structure PyramidRow :=
  (left : ℕ) (middle : ℕ) (right : ℕ)

/-- Represents the pyramid structure -/
structure Pyramid :=
  (top : ℕ)
  (second : PyramidRow)
  (third : PyramidRow)
  (bottom : PyramidRow)

/-- Checks if a pyramid is valid according to the multiplication rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.third.left = p.second.left * p.second.middle ∧
  p.third.middle = p.second.middle * p.second.right ∧
  p.third.right = p.second.right * p.bottom.right ∧
  p.top = p.second.left * p.second.right

theorem pyramid_theorem (p : Pyramid) 
  (h1 : p.second.left = 6)
  (h2 : p.third.left = 20)
  (h3 : p.bottom = PyramidRow.mk 20 30 72)
  (h4 : is_valid_pyramid p) : 
  p.top = 54 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_theorem_l1855_185519


namespace NUMINAMATH_CALUDE_equation_solution_l1855_185568

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) ↔ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1855_185568


namespace NUMINAMATH_CALUDE_crackers_distribution_l1855_185594

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_person : ℕ) :
  total_crackers = 22 →
  num_friends = 11 →
  crackers_per_person = total_crackers / num_friends →
  crackers_per_person = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l1855_185594


namespace NUMINAMATH_CALUDE_closest_point_to_cheese_l1855_185557

/-- The point where the mouse starts getting farther from the cheese -/
def closest_point : ℚ × ℚ := (3/17, 141/17)

/-- The location of the cheese -/
def cheese_location : ℚ × ℚ := (15, 12)

/-- The initial location of the mouse -/
def mouse_initial : ℚ × ℚ := (3, -3)

/-- The path of the mouse -/
def mouse_path (x : ℚ) : ℚ := -4 * x + 9

theorem closest_point_to_cheese :
  let (a, b) := closest_point
  (∀ x : ℚ, (x - 15)^2 + (mouse_path x - 12)^2 ≥ (a - 15)^2 + (b - 12)^2) ∧
  mouse_path a = b ∧
  a + b = 144/17 :=
sorry

end NUMINAMATH_CALUDE_closest_point_to_cheese_l1855_185557


namespace NUMINAMATH_CALUDE_student_calculation_difference_l1855_185512

/-- Proves that dividing a number by 4/5 instead of multiplying it by 4/5 results in a specific difference -/
theorem student_calculation_difference (number : ℝ) (h : number = 40.000000000000014) :
  (number / (4/5)) - (number * (4/5)) = 18.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l1855_185512


namespace NUMINAMATH_CALUDE_sixth_root_of_two_squared_equals_cube_root_of_two_l1855_185543

theorem sixth_root_of_two_squared_equals_cube_root_of_two : 
  (2^2)^(1/6) = 2^(1/3) := by sorry

end NUMINAMATH_CALUDE_sixth_root_of_two_squared_equals_cube_root_of_two_l1855_185543


namespace NUMINAMATH_CALUDE_product_of_decimals_l1855_185539

theorem product_of_decimals : (0.05 : ℝ) * 0.3 * 2 = 0.03 := by sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1855_185539


namespace NUMINAMATH_CALUDE_equation_roots_imply_m_range_l1855_185572

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    4^x₁ - m * 2^(x₁ + 1) + 2 - m = 0 ∧
    4^x₂ - m * 2^(x₂ + 1) + 2 - m = 0) →
  1 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_imply_m_range_l1855_185572


namespace NUMINAMATH_CALUDE_juan_number_operations_l1855_185529

theorem juan_number_operations (n : ℝ) : 
  (((n + 3) * 2 - 2) / 2 = 9) → (n = 7) := by
  sorry

end NUMINAMATH_CALUDE_juan_number_operations_l1855_185529


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1855_185573

theorem fraction_sum_equality : ∃ (a b c d e f : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (a ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   b ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   c ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   d ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   e ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ) ∧
   f ∈ ({1, 2, 3, 5, 6, 7} : Set ℕ)) ∧
  (Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1) ∧
  (a * d * f + c * b * f = e * b * d) ∧
  (b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1855_185573


namespace NUMINAMATH_CALUDE_equation_solution_l1855_185535

theorem equation_solution : ∃ x : ℝ, 
  ((3^2 - 5) / (0.08 * 7 + 2)) + Real.sqrt x = 10 ∧ x = 71.2715625 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1855_185535


namespace NUMINAMATH_CALUDE_train_journey_time_l1855_185522

/-- If a train travels at 4/7 of its usual speed and arrives 9 minutes late, 
    its usual time to cover the journey is 12 minutes. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
    (4 / 7 * usual_speed) * (usual_time + 9) = usual_speed * usual_time → 
    usual_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l1855_185522


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1855_185552

/-- Given a line segment with midpoint (-3, 2) and one endpoint (-7, 6), 
    prove that the other endpoint is (1, -2). -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (-3, 2) → endpoint1 = (-7, 6) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1855_185552


namespace NUMINAMATH_CALUDE_min_cost_2009_proof_l1855_185527

/-- Represents the available coin denominations in rubles -/
inductive Coin : Type
  | one : Coin
  | two : Coin
  | five : Coin
  | ten : Coin

/-- The value of a coin in rubles -/
def coin_value : Coin → ℕ
  | Coin.one => 1
  | Coin.two => 2
  | Coin.five => 5
  | Coin.ten => 10

/-- An arithmetic expression using coins and operations -/
inductive Expr : Type
  | coin : Coin → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an expression to its numeric value -/
def eval : Expr → ℕ
  | Expr.coin c => coin_value c
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Calculates the total cost of an expression in rubles -/
def cost : Expr → ℕ
  | Expr.coin c => coin_value c
  | Expr.add e1 e2 => cost e1 + cost e2
  | Expr.sub e1 e2 => cost e1 + cost e2
  | Expr.mul e1 e2 => cost e1 + cost e2
  | Expr.div e1 e2 => cost e1 + cost e2

/-- The minimum cost to create an expression equal to 2009 -/
def min_cost_2009 : ℕ := 23

theorem min_cost_2009_proof :
  ∀ e : Expr, eval e = 2009 → cost e ≥ min_cost_2009 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_2009_proof_l1855_185527


namespace NUMINAMATH_CALUDE_problem_solution_l1855_185533

def f (a : ℝ) (x : ℝ) : ℝ := |3*x - a| - 2*|x - 1|

theorem problem_solution :
  (∀ x : ℝ, f (-3) x > 1 ↔ (x < -6 ∨ x > 1)) ∧
  (∃ x : ℝ, f a x ≥ 6 + |x - 1| → (a ≥ 9 ∨ a < -3)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1855_185533


namespace NUMINAMATH_CALUDE_oreo_cheesecake_graham_crackers_l1855_185599

theorem oreo_cheesecake_graham_crackers :
  ∀ (G : ℕ) (oreos : ℕ),
  oreos = 15 →
  (∃ (cheesecakes : ℕ),
    cheesecakes * 2 = G - 4 ∧
    cheesecakes * 3 ≤ oreos ∧
    ∀ (c : ℕ), c * 2 ≤ G - 4 ∧ c * 3 ≤ oreos → c ≤ cheesecakes) →
  G = 14 := by sorry

end NUMINAMATH_CALUDE_oreo_cheesecake_graham_crackers_l1855_185599


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l1855_185544

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation x^2 - x + 3 = 0 -/
def given_equation : QuadraticEquation := { a := 1, b := -1, c := 3 }

theorem coefficients_of_given_equation :
  given_equation.a = 1 ∧ given_equation.b = -1 ∧ given_equation.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l1855_185544


namespace NUMINAMATH_CALUDE_permutation_count_l1855_185547

/-- The number of X's in the original string -/
def num_X : ℕ := 4

/-- The number of Y's in the original string -/
def num_Y : ℕ := 5

/-- The number of Z's in the original string -/
def num_Z : ℕ := 9

/-- The total length of the string -/
def total_length : ℕ := num_X + num_Y + num_Z

/-- The length of the first section where X is not allowed -/
def first_section : ℕ := 5

/-- The length of the middle section where Y is not allowed -/
def middle_section : ℕ := 6

/-- The length of the last section where Z is not allowed -/
def last_section : ℕ := 7

/-- The number of permutations satisfying the given conditions -/
def M : ℕ := sorry

theorem permutation_count : M % 1000 = 30 := by sorry

end NUMINAMATH_CALUDE_permutation_count_l1855_185547


namespace NUMINAMATH_CALUDE_nephews_difference_l1855_185553

theorem nephews_difference (alden_past : ℕ) (total : ℕ) : 
  alden_past = 50 →
  total = 260 →
  ∃ (alden_now vihaan : ℕ),
    alden_now = 2 * alden_past ∧
    vihaan > alden_now ∧
    alden_now + vihaan = total ∧
    vihaan - alden_now = 60 :=
by sorry

end NUMINAMATH_CALUDE_nephews_difference_l1855_185553


namespace NUMINAMATH_CALUDE_meeting_participants_count_l1855_185523

theorem meeting_participants_count : 
  ∀ (F M : ℕ),
  F = 330 →
  (F / 2 : ℚ) = 165 →
  (F + M) / 3 = F / 2 + M / 4 →
  F + M = 990 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_count_l1855_185523


namespace NUMINAMATH_CALUDE_total_songs_bought_l1855_185550

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of albums Megan bought -/
def total_albums : ℕ := country_albums + pop_albums

/-- Theorem: The total number of songs Megan bought is 70 -/
theorem total_songs_bought : total_albums * songs_per_album = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_songs_bought_l1855_185550


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1855_185507

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) 
  (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1855_185507


namespace NUMINAMATH_CALUDE_symmetry_implies_difference_l1855_185560

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_difference (a b : ℝ) :
  symmetric_wrt_y_axis (a, 3) (4, b) → a - b = -7 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_implies_difference_l1855_185560


namespace NUMINAMATH_CALUDE_arrangements_count_l1855_185561

/-- The number of candidates -/
def total_candidates : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The number of students who can be assigned to swimming -/
def swimming_candidates : ℕ := total_candidates - 1

/-- The number of different arrangements -/
def arrangements : ℕ := swimming_candidates * (total_candidates - 1) * (total_candidates - 2)

theorem arrangements_count : arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1855_185561
