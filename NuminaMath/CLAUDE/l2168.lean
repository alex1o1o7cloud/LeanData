import Mathlib

namespace NUMINAMATH_CALUDE_probability_heart_spade_queen_value_l2168_216897

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (valid : cards.card = 52)

/-- Represents a suit in a deck of cards -/
inductive Suit
| Hearts | Spades | Diamonds | Clubs

/-- Represents a rank in a deck of cards -/
inductive Rank
| Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

/-- A function to check if a card is a heart -/
def is_heart (card : Nat × Nat) : Prop := sorry

/-- A function to check if a card is a spade -/
def is_spade (card : Nat × Nat) : Prop := sorry

/-- A function to check if a card is a queen -/
def is_queen (card : Nat × Nat) : Prop := sorry

/-- The probability of drawing a heart first, a spade second, and a queen third -/
def probability_heart_spade_queen (d : Deck) : ℚ := sorry

/-- Theorem stating the probability of drawing a heart first, a spade second, and a queen third -/
theorem probability_heart_spade_queen_value (d : Deck) : 
  probability_heart_spade_queen d = 221 / 44200 := by sorry

end NUMINAMATH_CALUDE_probability_heart_spade_queen_value_l2168_216897


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2168_216864

/-- Given that the solution set of ax^2 + 5x + b > 0 is {x | 2 < x < 3},
    prove that the solution set of bx^2 - 5x + a < 0 is {x | x < -1/2 or x > -1/3} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | a * x^2 + 5 * x + b > 0}) :
  {x : ℝ | b * x^2 - 5 * x + a < 0} = {x : ℝ | x < -1/2 ∨ x > -1/3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2168_216864


namespace NUMINAMATH_CALUDE_parallel_lines_min_value_l2168_216890

theorem parallel_lines_min_value (m n : ℕ+) : 
  (∀ x y : ℝ, x + (n.val - 1) * y - 2 = 0 ↔ m.val * x + y + 3 = 0) →
  (∀ k : ℕ+, 2 * m.val + n.val ≤ k.val → k.val = 11) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_min_value_l2168_216890


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2168_216840

theorem quadratic_equation_real_roots (m n : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁^2 - (m + n)*x₁ + m*n = 0 ∧ x₂^2 - (m + n)*x₂ + m*n = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2168_216840


namespace NUMINAMATH_CALUDE_women_count_correct_l2168_216849

/-- The number of women working with men to complete a job -/
def num_women : ℕ := 15

/-- The number of men working on the job -/
def num_men : ℕ := 10

/-- The number of days it takes for the group to complete the job -/
def group_days : ℕ := 6

/-- The number of days it takes for one man to complete the job -/
def man_days : ℕ := 100

/-- The number of days it takes for one woman to complete the job -/
def woman_days : ℕ := 225

/-- Theorem stating that the number of women working with the men is correct -/
theorem women_count_correct :
  (num_men : ℚ) / man_days + (num_women : ℚ) / woman_days = 1 / group_days :=
sorry


end NUMINAMATH_CALUDE_women_count_correct_l2168_216849


namespace NUMINAMATH_CALUDE_range_when_a_is_one_a_values_for_all_x_geq_one_l2168_216861

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

-- Theorem for part I
theorem range_when_a_is_one :
  Set.range (fun x => f x 1) = Set.Ici 5 := by sorry

-- Theorem for part II
theorem a_values_for_all_x_geq_one :
  {a : ℝ | ∀ x, f x a ≥ 1} = Set.Iic (-5) ∪ Set.Ici (-3) := by sorry

end NUMINAMATH_CALUDE_range_when_a_is_one_a_values_for_all_x_geq_one_l2168_216861


namespace NUMINAMATH_CALUDE_unicycle_count_l2168_216899

theorem unicycle_count :
  ∀ (num_bicycles num_tricycles num_unicycles : ℕ),
    num_bicycles = 3 →
    num_tricycles = 4 →
    num_bicycles * 2 + num_tricycles * 3 + num_unicycles * 1 = 25 →
    num_unicycles = 7 := by
  sorry

end NUMINAMATH_CALUDE_unicycle_count_l2168_216899


namespace NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l2168_216872

-- Define the structure for a tile
structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

-- Define the tiles
def tileI : Tile := ⟨1, 2, 5, 6⟩
def tileII : Tile := ⟨6, 3, 1, 5⟩
def tileIII : Tile := ⟨5, 7, 2, 3⟩
def tileIV : Tile := ⟨3, 5, 7, 2⟩

-- Define a function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) (side : String) : Prop :=
  match side with
  | "right" => t1.right = t2.left
  | "left" => t1.left = t2.right
  | "top" => t1.top = t2.bottom
  | "bottom" => t1.bottom = t2.top
  | _ => False

-- Theorem stating that Tile IV is the only tile that can be placed in Rectangle C
theorem tileIV_in_rectangle_C :
  (canBeAdjacent tileIV tileIII "left") ∧
  (¬ canBeAdjacent tileI tileIII "left") ∧
  (¬ canBeAdjacent tileII tileIII "left") ∧
  (∃ (t : Tile), t = tileIV ∧ canBeAdjacent t tileIII "left") :=
sorry

end NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l2168_216872


namespace NUMINAMATH_CALUDE_tessellation_theorem_l2168_216828

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  interiorAngle : ℝ

/-- Checks if two regular polygons can tessellate -/
def canTessellate (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n1 n2 : ℕ), n1 * p1.interiorAngle + n2 * p2.interiorAngle = 360

theorem tessellation_theorem :
  let triangle : RegularPolygon := ⟨3, 60⟩
  let square : RegularPolygon := ⟨4, 90⟩
  let hexagon : RegularPolygon := ⟨6, 120⟩
  let octagon : RegularPolygon := ⟨8, 135⟩

  (canTessellate triangle square) ∧
  (canTessellate triangle hexagon) ∧
  (canTessellate octagon square) ∧
  ¬(canTessellate hexagon square) :=
by sorry

end NUMINAMATH_CALUDE_tessellation_theorem_l2168_216828


namespace NUMINAMATH_CALUDE_first_week_cases_l2168_216896

/-- Given the number of coronavirus cases in New York over three weeks,
    prove that the number of cases in the first week was 3750. -/
theorem first_week_cases (first_week : ℕ) : 
  (first_week + first_week / 2 + (first_week / 2 + 2000) = 9500) → 
  first_week = 3750 := by
  sorry

end NUMINAMATH_CALUDE_first_week_cases_l2168_216896


namespace NUMINAMATH_CALUDE_chord_length_of_concentric_circles_l2168_216812

/-- Given two concentric circles with the following properties:
  - The area of the ring between the circles is 50π/3 square inches
  - The diameter of the larger circle is 10 inches
  This theorem proves that the length of a chord of the larger circle 
  that is tangent to the smaller circle is 10√6/3 inches. -/
theorem chord_length_of_concentric_circles (a b : ℝ) : 
  a = 5 →  -- Radius of larger circle
  π * a^2 - π * b^2 = (50/3) * π →  -- Area of ring
  ∃ c : ℝ, c = (10 * Real.sqrt 6) / 3 ∧ 
    c^2 = 4 * (a^2 - b^2) :=  -- Length of chord tangent to smaller circle
by
  sorry

#check chord_length_of_concentric_circles

end NUMINAMATH_CALUDE_chord_length_of_concentric_circles_l2168_216812


namespace NUMINAMATH_CALUDE_milk_fraction_after_pours_l2168_216801

/-- Represents the contents of a cup -/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Represents the state of both cups -/
structure TwoCapState where
  cup1 : CupContents
  cup2 : CupContents

/-- Initial state of the cups -/
def initial_state : TwoCapState :=
  { cup1 := { tea := 6, milk := 0 },
    cup2 := { tea := 0, milk := 6 } }

/-- Pour one-third of tea from cup1 to cup2 -/
def pour_tea (state : TwoCapState) : TwoCapState := sorry

/-- Pour half of the mixture from cup2 to cup1 -/
def pour_mixture (state : TwoCapState) : TwoCapState := sorry

/-- Calculate the fraction of milk in a cup -/
def milk_fraction (cup : CupContents) : ℚ := sorry

/-- The main theorem to prove -/
theorem milk_fraction_after_pours :
  let state1 := pour_tea initial_state
  let state2 := pour_mixture state1
  milk_fraction state2.cup1 = 3/8 := by sorry

end NUMINAMATH_CALUDE_milk_fraction_after_pours_l2168_216801


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_intersection_angle_l2168_216839

/-- A regular pentagon is a polygon with 5 equal sides and 5 equal angles. -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The diagonals of a pentagon are line segments connecting non-adjacent vertices. -/
def diagonal (p : RegularPentagon) (i j : Fin 5) : sorry := sorry

/-- The intersection point of two diagonals in a pentagon. -/
def intersectionPoint (p : RegularPentagon) (d1 d2 : sorry) : ℝ × ℝ := sorry

/-- The angle between two line segments at their intersection point. -/
def angleBetween (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem regular_pentagon_diagonal_intersection_angle (p : RegularPentagon) :
  let s := intersectionPoint p (diagonal p 0 2) (diagonal p 1 3)
  angleBetween (p.vertices 2) s (p.vertices 3) = 72 := by sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_intersection_angle_l2168_216839


namespace NUMINAMATH_CALUDE_systematic_sampling_third_selection_l2168_216843

theorem systematic_sampling_third_selection
  (total_students : ℕ)
  (selected_students : ℕ)
  (first_selection : ℕ)
  (h1 : total_students = 100)
  (h2 : selected_students = 10)
  (h3 : first_selection = 3)
  : (first_selection + 2 * (total_students / selected_students)) % 100 = 23 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_third_selection_l2168_216843


namespace NUMINAMATH_CALUDE_inner_quadrilateral_area_l2168_216895

/-- A square with side length 10 cm, partitioned by lines from corners to opposite midpoints -/
structure PartitionedSquare where
  side_length : ℝ
  is_ten_cm : side_length = 10

/-- The inner quadrilateral formed by the intersecting lines -/
def inner_quadrilateral (s : PartitionedSquare) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The area of the inner quadrilateral is 25 cm² -/
theorem inner_quadrilateral_area (s : PartitionedSquare) :
  area (inner_quadrilateral s) = 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_quadrilateral_area_l2168_216895


namespace NUMINAMATH_CALUDE_point_coordinates_on_horizontal_line_l2168_216818

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line parallel to the x-axis -/
structure HorizontalLine where
  y : ℝ

def Point.liesOn (p : Point) (l : HorizontalLine) : Prop :=
  p.y = l.y

theorem point_coordinates_on_horizontal_line 
  (m : ℝ)
  (P : Point)
  (A : Point)
  (l : HorizontalLine)
  (h1 : P = ⟨2*m + 4, m - 1⟩)
  (h2 : A = ⟨2, -4⟩)
  (h3 : l.y = A.y)
  (h4 : P.liesOn l) :
  P = ⟨-2, -4⟩ :=
sorry

end NUMINAMATH_CALUDE_point_coordinates_on_horizontal_line_l2168_216818


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l2168_216827

def n : ℕ := 2^5 * 3^5 * 5^4 * 7^6

theorem smallest_n_for_integer_roots :
  (∃ (a b c : ℕ), (5 * n = a^5) ∧ (6 * n = b^6) ∧ (7 * n = c^7)) ∧
  (∀ m : ℕ, m < n → ¬(∃ (x y z : ℕ), (5 * m = x^5) ∧ (6 * m = y^6) ∧ (7 * m = z^7))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_roots_l2168_216827


namespace NUMINAMATH_CALUDE_expected_value_of_x_l2168_216842

/-- Represents the contingency table data -/
structure ContingencyTable where
  boys_a : ℕ
  boys_b : ℕ
  girls_a : ℕ
  girls_b : ℕ

/-- Represents the distribution of X -/
structure Distribution where
  p0 : ℚ
  p1 : ℚ
  p2 : ℚ
  p3 : ℚ

/-- Main theorem statement -/
theorem expected_value_of_x (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ) 
  (table : ContingencyTable) (dist : Distribution) : 
  total_students = 450 →
  total_boys = 250 →
  total_girls = 200 →
  table.boys_a + table.boys_b = total_boys →
  table.girls_a + table.girls_b = total_girls →
  table.boys_b = 150 →
  table.girls_a = 50 →
  dist.p0 = 1/6 →
  dist.p1 = 1/2 →
  dist.p2 = 3/10 →
  dist.p3 = 1/30 →
  0 * dist.p0 + 1 * dist.p1 + 2 * dist.p2 + 3 * dist.p3 = 6/5 := by
  sorry


end NUMINAMATH_CALUDE_expected_value_of_x_l2168_216842


namespace NUMINAMATH_CALUDE_circle_area_difference_l2168_216829

theorem circle_area_difference (r₁ r₂ d : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 15) (h₃ : d = 8) 
  (h₄ : d = r₁ + r₂) : π * r₂^2 - π * r₁^2 = 200 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2168_216829


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l2168_216841

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l2168_216841


namespace NUMINAMATH_CALUDE_tank_capacity_l2168_216891

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 9 →
  final_fraction = 9/10 →
  ∃ (capacity : Rat), capacity = 60 ∧
    final_fraction * capacity - initial_fraction * capacity = added_amount :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l2168_216891


namespace NUMINAMATH_CALUDE_amy_homework_time_l2168_216835

/-- Calculates the total time needed to complete homework with breaks -/
def total_homework_time (math_problems : ℕ) (spelling_problems : ℕ) 
  (math_rate : ℕ) (spelling_rate : ℕ) (break_duration : ℚ) : ℚ :=
  let work_hours : ℚ := (math_problems / math_rate + spelling_problems / spelling_rate : ℚ)
  let break_hours : ℚ := (work_hours.floor - 1) * break_duration
  work_hours + break_hours

/-- Theorem: Amy will take 11 hours to finish her homework -/
theorem amy_homework_time : 
  total_homework_time 18 6 3 2 (1/4) = 11 := by sorry

end NUMINAMATH_CALUDE_amy_homework_time_l2168_216835


namespace NUMINAMATH_CALUDE_polynomial_solution_set_l2168_216831

theorem polynomial_solution_set : ∃ (S : Set ℂ), 
  S = {z : ℂ | z^4 + 2*z^3 + 2*z^2 + 2*z + 1 = 0} ∧ 
  S = {-1, Complex.I, -Complex.I} := by
  sorry

end NUMINAMATH_CALUDE_polynomial_solution_set_l2168_216831


namespace NUMINAMATH_CALUDE_plane_line_relations_l2168_216870

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem plane_line_relations
  (α β : Plane) (l m : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : l ≠ m)
  (h_l_perp_α : perpendicular l α)
  (h_m_subset_β : subset m β) :
  (parallel α β → line_perpendicular l m) ∧
  (perpendicular l β → line_parallel m α) :=
sorry

end NUMINAMATH_CALUDE_plane_line_relations_l2168_216870


namespace NUMINAMATH_CALUDE_negative_square_times_a_l2168_216847

theorem negative_square_times_a (a : ℝ) : -a^2 * a = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_times_a_l2168_216847


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2168_216848

/-- Proves that the cost price of an article is 350, given the selling price and profit percentage. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 455 → profit_percentage = 30 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 350 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2168_216848


namespace NUMINAMATH_CALUDE_certain_number_problem_l2168_216852

theorem certain_number_problem : ∃ x : ℕ, 
  220025 = (x + 445) * (2 * (x - 445)) + 25 ∧ 
  x = 555 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2168_216852


namespace NUMINAMATH_CALUDE_rainfall_rate_calculation_l2168_216862

/-- Proves that the rainfall rate is 5 cm/hour given the specified conditions -/
theorem rainfall_rate_calculation (depth : ℝ) (area : ℝ) (time : ℝ) 
  (h_depth : depth = 15)
  (h_area : area = 300)
  (h_time : time = 3) :
  (depth * area) / (time * area) = 5 := by
sorry

end NUMINAMATH_CALUDE_rainfall_rate_calculation_l2168_216862


namespace NUMINAMATH_CALUDE_curve_symmetry_about_origin_l2168_216869

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop := 3 * x^2 - 8 * x * y + 2 * y^2 = 0

-- Theorem stating the symmetry about the origin
theorem curve_symmetry_about_origin :
  ∀ (x y : ℝ), curve_equation x y ↔ curve_equation (-x) (-y) :=
by sorry

end NUMINAMATH_CALUDE_curve_symmetry_about_origin_l2168_216869


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l2168_216819

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (1 - Real.sin θ)

-- Define the Cartesian equation derived from the polar equation
def cartesian_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 0

-- Theorem stating that the equation represents a circle (point)
theorem polar_equation_is_circle :
  ∃! (x y : ℝ), cartesian_equation x y ∧ x = 0 ∧ y = 1 :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l2168_216819


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2168_216854

/-- The number of toppings available for sandwiches -/
def num_toppings : ℕ := 9

/-- The number of patty choices available for sandwiches -/
def num_patties : ℕ := 2

/-- The total number of different sandwich combinations -/
def total_combinations : ℕ := 2^num_toppings * num_patties

theorem sandwich_combinations :
  total_combinations = 1024 :=
sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2168_216854


namespace NUMINAMATH_CALUDE_ab_greater_than_sum_l2168_216893

theorem ab_greater_than_sum (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_sum_l2168_216893


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l2168_216804

/-- A triangle with vertices A(2,3), B(-4,1), and C(5,-6) -/
structure Triangle where
  A : ℝ × ℝ := (2, 3)
  B : ℝ × ℝ := (-4, 1)
  C : ℝ × ℝ := (5, -6)

/-- The equation of an angle bisector in the form 3x + by + c = 0 -/
structure AngleBisectorEquation where
  b : ℝ
  c : ℝ

/-- The angle bisector of ∠A in the given triangle -/
def angleBisectorA (t : Triangle) : AngleBisectorEquation :=
  sorry

theorem angle_bisector_sum (t : Triangle) :
  let bisector := angleBisectorA t
  bisector.b + bisector.c = -2 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l2168_216804


namespace NUMINAMATH_CALUDE_inequality_holds_iff_x_in_range_l2168_216811

theorem inequality_holds_iff_x_in_range :
  ∀ x : ℝ, (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ 
  (x < -1 ∨ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_x_in_range_l2168_216811


namespace NUMINAMATH_CALUDE_angelinas_speed_l2168_216878

theorem angelinas_speed (v : ℝ) 
  (home_to_grocery : 840 / v = 510 / (1.5 * v) + 40)
  (grocery_to_library : 510 / (1.5 * v) = 480 / (2 * v) + 20) :
  2 * v = 25 := by
  sorry

end NUMINAMATH_CALUDE_angelinas_speed_l2168_216878


namespace NUMINAMATH_CALUDE_payment_sequence_aperiodic_l2168_216887

/-- A sequence of daily payments (1 or 2 rubles) -/
def PaymentSequence := ℕ → Fin 2

/-- The sum of the first n payments -/
def TotalPayment (seq : PaymentSequence) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => seq i + 1)

/-- A payment sequence is valid if the total payment is always the nearest integer to n√2 -/
def IsValidPaymentSequence (seq : PaymentSequence) : Prop :=
  ∀ n : ℕ, |TotalPayment seq n - n * Real.sqrt 2| ≤ 1/2

/-- A sequence is periodic if it repeats after some point -/
def IsPeriodic (seq : PaymentSequence) : Prop :=
  ∃ (N T : ℕ), T > 0 ∧ ∀ n ≥ N, seq (n + T) = seq n

theorem payment_sequence_aperiodic (seq : PaymentSequence) 
  (hvalid : IsValidPaymentSequence seq) : ¬IsPeriodic seq := by
  sorry

end NUMINAMATH_CALUDE_payment_sequence_aperiodic_l2168_216887


namespace NUMINAMATH_CALUDE_ted_blue_mushrooms_l2168_216879

theorem ted_blue_mushrooms :
  let bill_red : ℕ := 12
  let bill_brown : ℕ := 6
  let ted_green : ℕ := 14
  let ted_blue : ℕ := x
  let white_spotted_total : ℕ := 17
  let white_spotted_bill_red : ℕ := bill_red * 2 / 3
  let white_spotted_bill_brown : ℕ := bill_brown
  let white_spotted_ted_blue : ℕ := ted_blue / 2
  white_spotted_total = white_spotted_bill_red + white_spotted_bill_brown + white_spotted_ted_blue →
  ted_blue = 10 := by
sorry

end NUMINAMATH_CALUDE_ted_blue_mushrooms_l2168_216879


namespace NUMINAMATH_CALUDE_condition_analysis_l2168_216845

theorem condition_analysis (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * b^2 > 0 → a > b) ∧
  (∃ a b : ℝ, a > b ∧ (a - b) * b^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_analysis_l2168_216845


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l2168_216873

theorem consecutive_odd_numbers (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →              -- b is the next odd number after a
  c = a + 4 →              -- c is the third odd number
  d = a + 6 →              -- d is the fourth odd number
  e = a + 8 →              -- e is the fifth odd number
  a + c = 146 →            -- sum of a and c is 146
  e = 79 := by             -- prove that e equals 79
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l2168_216873


namespace NUMINAMATH_CALUDE_scissors_count_l2168_216820

/-- The total number of scissors after adding more to an initial amount -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 54 initial scissors and 22 added scissors, the total is 76 -/
theorem scissors_count : total_scissors 54 22 = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l2168_216820


namespace NUMINAMATH_CALUDE_jared_tom_age_ratio_l2168_216813

theorem jared_tom_age_ratio : 
  ∀ (tom_future_age jared_current_age : ℕ),
    tom_future_age = 30 →
    jared_current_age = 48 →
    ∃ (jared_past_age tom_past_age : ℕ),
      jared_past_age = jared_current_age - 2 ∧
      tom_past_age = tom_future_age - 7 ∧
      jared_past_age = 2 * tom_past_age :=
by sorry

end NUMINAMATH_CALUDE_jared_tom_age_ratio_l2168_216813


namespace NUMINAMATH_CALUDE_triples_satisfying_equation_l2168_216859

theorem triples_satisfying_equation : 
  ∀ (a b p : ℕ), 
    0 < a ∧ 0 < b ∧ 0 < p ∧ 
    Nat.Prime p ∧
    a^p - b^p = 2013 →
    ((a = 337 ∧ b = 334 ∧ p = 2) ∨ 
     (a = 97 ∧ b = 86 ∧ p = 2) ∨ 
     (a = 47 ∧ b = 14 ∧ p = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triples_satisfying_equation_l2168_216859


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l2168_216822

theorem intersection_sum_zero (α β : ℝ) : 
  (∃ x₀ : ℝ, 
    (x₀ / (Real.sin α + Real.sin β) + (-x₀) / (Real.sin α + Real.cos β) = 1) ∧
    (x₀ / (Real.cos α + Real.sin β) + (-x₀) / (Real.cos α + Real.cos β) = 1)) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l2168_216822


namespace NUMINAMATH_CALUDE_maria_coin_count_l2168_216802

theorem maria_coin_count (num_stacks : ℕ) (coins_per_stack : ℕ) : 
  num_stacks = 5 → coins_per_stack = 3 → num_stacks * coins_per_stack = 15 := by
  sorry

end NUMINAMATH_CALUDE_maria_coin_count_l2168_216802


namespace NUMINAMATH_CALUDE_range_of_a_and_m_l2168_216885

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- Define the theorem
theorem range_of_a_and_m (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_and_m_l2168_216885


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_l2168_216800

theorem reciprocal_sum_one (x y z : ℕ+) : 
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1 ↔ 
  ((x = 2 ∧ y = 4 ∧ z = 4) ∨ 
   (x = 2 ∧ y = 3 ∧ z = 6) ∨ 
   (x = 3 ∧ y = 3 ∧ z = 3)) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_one_l2168_216800


namespace NUMINAMATH_CALUDE_inequality_proof_l2168_216853

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + a*b + b^2 = 3*c^2) (h2 : a^3 + a^2*b + a*b^2 + b^3 = 4*d^3) :
  a + b + d ≤ 3*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2168_216853


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_36_l2168_216894

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = x * 1000 + 410 + y

theorem four_digit_divisible_by_36 :
  ∀ n : ℕ, is_valid_number n ∧ n % 36 = 0 ↔ n = 2412 ∨ n = 7416 := by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_36_l2168_216894


namespace NUMINAMATH_CALUDE_grassy_width_is_55_l2168_216855

/-- Represents the dimensions and cost of a rectangular plot with a gravel path -/
structure Plot where
  length : ℝ
  path_width : ℝ
  gravel_cost_per_sqm : ℝ
  total_gravel_cost : ℝ

/-- Calculates the width of the grassy area given the plot dimensions and gravel cost -/
def calculate_grassy_width (p : Plot) : ℝ :=
  sorry

/-- Theorem stating that for the given dimensions and cost, the grassy width is 55 meters -/
theorem grassy_width_is_55 (p : Plot) 
  (h1 : p.length = 110)
  (h2 : p.path_width = 2.5)
  (h3 : p.gravel_cost_per_sqm = 0.6)
  (h4 : p.total_gravel_cost = 510) :
  calculate_grassy_width p = 55 :=
sorry

end NUMINAMATH_CALUDE_grassy_width_is_55_l2168_216855


namespace NUMINAMATH_CALUDE_contemporary_probability_is_five_ninths_l2168_216806

/-- Represents a scientist with a birth year and lifespan -/
structure Scientist where
  birth_year : ℝ
  lifespan : ℝ

/-- The total time span in years -/
def total_span : ℝ := 600

/-- The probability that two scientists were contemporaries -/
noncomputable def contemporary_probability (s1 s2 : Scientist) : ℝ :=
  let overlap_area := (total_span - (s1.lifespan + s2.lifespan)) ^ 2
  (total_span ^ 2 - overlap_area) / (total_span ^ 2)

/-- The main theorem stating the probability of two scientists being contemporaries -/
theorem contemporary_probability_is_five_ninths :
  ∃ (s1 s2 : Scientist),
    s1.lifespan = 110 ∧
    s2.lifespan = 90 ∧
    s1.birth_year ≥ 0 ∧
    s1.birth_year ≤ total_span ∧
    s2.birth_year ≥ 0 ∧
    s2.birth_year ≤ total_span ∧
    contemporary_probability s1 s2 = 5 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_contemporary_probability_is_five_ninths_l2168_216806


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2168_216889

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2168_216889


namespace NUMINAMATH_CALUDE_prob_spade_then_king_value_l2168_216805

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a spade as the first card and a king as the second card -/
def prob_spade_then_king : ℚ :=
  (NumSpades / StandardDeck) * (NumKings / (StandardDeck - 1))

theorem prob_spade_then_king_value :
  prob_spade_then_king = 17 / 884 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_then_king_value_l2168_216805


namespace NUMINAMATH_CALUDE_solution_x_l2168_216857

theorem solution_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 104) : x = 34 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_l2168_216857


namespace NUMINAMATH_CALUDE_balloons_given_to_fred_l2168_216816

/-- Given that Tom initially had 30 balloons and now has 14 balloons,
    prove that he gave 16 balloons to Fred. -/
theorem balloons_given_to_fred 
  (initial_balloons : ℕ) 
  (remaining_balloons : ℕ) 
  (h1 : initial_balloons = 30) 
  (h2 : remaining_balloons = 14) : 
  initial_balloons - remaining_balloons = 16 :=
by sorry

end NUMINAMATH_CALUDE_balloons_given_to_fred_l2168_216816


namespace NUMINAMATH_CALUDE_find_divisor_l2168_216881

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 14698 →
  quotient = 89 →
  remainder = 14 →
  dividend = divisor * quotient + remainder →
  divisor = 165 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2168_216881


namespace NUMINAMATH_CALUDE_max_three_digit_divisible_by_15_existence_of_solution_l2168_216892

def is_valid_assignment (n a b c d e : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9 ∧ c ≥ 1 ∧ c ≤ 9 ∧ d ≥ 1 ∧ d ≤ 9 ∧ e ≥ 1 ∧ e ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  n / (a * b + c + d * e) = 15

theorem max_three_digit_divisible_by_15 :
  ∀ n a b c d e : ℕ,
    is_valid_assignment n a b c d e →
    n ≤ 975 :=
by sorry

theorem existence_of_solution :
  ∃ n a b c d e : ℕ,
    is_valid_assignment n a b c d e ∧
    n = 975 :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_divisible_by_15_existence_of_solution_l2168_216892


namespace NUMINAMATH_CALUDE_max_volume_container_l2168_216868

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : ℝ :=
  d.shortSide * d.longSide * d.height

/-- Represents the constraints of the problem --/
def isValidContainer (d : ContainerDimensions) : Prop :=
  d.longSide = d.shortSide + 0.5 ∧
  2 * (d.shortSide + d.longSide) + 4 * d.height = 14.8 ∧
  d.shortSide > 0 ∧ d.longSide > 0 ∧ d.height > 0

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    isValidContainer d ∧
    volume d = 1.8 ∧
    d.height = 1.2 ∧
    ∀ (d' : ContainerDimensions), isValidContainer d' → volume d' ≤ volume d :=
by sorry

end NUMINAMATH_CALUDE_max_volume_container_l2168_216868


namespace NUMINAMATH_CALUDE_video_recorder_markup_percentage_l2168_216834

/-- Proves that the markup percentage is 20% given the problem conditions -/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_price : ℝ)
  (employee_discount : ℝ)
  (markup_percentage : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_price = 192)
  (h3 : employee_discount = 0.20)
  (h4 : employee_price = (1 - employee_discount) * (wholesale_cost * (1 + markup_percentage / 100)))
  : markup_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_markup_percentage_l2168_216834


namespace NUMINAMATH_CALUDE_probability_sum_four_two_dice_l2168_216844

theorem probability_sum_four_two_dice : 
  let dice_count : ℕ := 2
  let faces_per_die : ℕ := 6
  let target_sum : ℕ := 4
  let total_outcomes : ℕ := faces_per_die ^ dice_count
  let favorable_outcomes : ℕ := 3
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_four_two_dice_l2168_216844


namespace NUMINAMATH_CALUDE_stratified_sample_for_model_a_l2168_216883

/-- Calculates the number of items to be selected in a stratified sample -/
def stratified_sample_size (model_volume : ℕ) (total_volume : ℕ) (total_sample : ℕ) : ℕ :=
  (model_volume * total_sample) / total_volume

theorem stratified_sample_for_model_a 
  (volume_a volume_b volume_c total_sample : ℕ) 
  (h_positive : volume_a > 0 ∧ volume_b > 0 ∧ volume_c > 0 ∧ total_sample > 0) :
  stratified_sample_size volume_a (volume_a + volume_b + volume_c) total_sample = 
    (volume_a * total_sample) / (volume_a + volume_b + volume_c) :=
by
  sorry

#eval stratified_sample_size 1200 9200 46

end NUMINAMATH_CALUDE_stratified_sample_for_model_a_l2168_216883


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l2168_216856

theorem opposite_of_negative_three :
  -((-3 : ℤ)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l2168_216856


namespace NUMINAMATH_CALUDE_monotonicity_of_g_no_solutions_for_equation_l2168_216823

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a
def g (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / (x + 1)

theorem monotonicity_of_g :
  a = 1 →
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 - 1 → g a x₁ < g a x₂) ∧
  (∀ x₁ x₂, Real.exp 1 - 1 < x₁ ∧ x₁ < x₂ → g a x₁ > g a x₂) :=
sorry

theorem no_solutions_for_equation :
  0 < a → a < 2/3 → ∀ x, f a x ≠ (x + 1) * g a x :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_g_no_solutions_for_equation_l2168_216823


namespace NUMINAMATH_CALUDE_expression_equality_l2168_216832

theorem expression_equality : 
  |1 - Real.sqrt 2| - 2 * Real.cos (45 * π / 180) + (1 / 2)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2168_216832


namespace NUMINAMATH_CALUDE_average_problem_l2168_216836

theorem average_problem (x y : ℝ) : 
  ((100 + 200300 + x) / 3 = 250) → 
  ((300 + 150100 + x + y) / 4 = 200) → 
  y = -4250 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l2168_216836


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2168_216808

/-- Calculates the distance between two cars on a main road -/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - car1_distance - car2_distance

theorem car_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 62) :
  distance_between_cars initial_distance car1_distance car2_distance = 38 := by
  sorry

#eval distance_between_cars 150 50 62

end NUMINAMATH_CALUDE_car_distance_theorem_l2168_216808


namespace NUMINAMATH_CALUDE_tangent_points_coordinates_fixed_points_on_circle_l2168_216850

/-- Circle M with equation x^2 + (y-2)^2 = 1 -/
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

/-- Line l with equation x - 2y = 0 -/
def line_l (x y : ℝ) : Prop := x - 2*y = 0

/-- Point P lies on line l -/
def P_on_line_l (x y : ℝ) : Prop := line_l x y

/-- PA and PB are tangents to circle M -/
def tangents_to_M (xp yp xa ya xb yb : ℝ) : Prop :=
  circle_M xa ya ∧ circle_M xb yb ∧
  ((xp - xa) * xa + (yp - ya) * (ya - 2) = 0) ∧
  ((xp - xb) * xb + (yp - yb) * (yb - 2) = 0)

/-- Angle APB is 60 degrees -/
def angle_APB_60 (xp yp xa ya xb yb : ℝ) : Prop :=
  let v1x := xa - xp
  let v1y := ya - yp
  let v2x := xb - xp
  let v2y := yb - yp
  (v1x * v2x + v1y * v2y)^2 = 3 * ((v1x^2 + v1y^2) * (v2x^2 + v2y^2)) / 4

theorem tangent_points_coordinates :
  ∀ (xp yp xa ya xb yb : ℝ),
  P_on_line_l xp yp →
  tangents_to_M xp yp xa ya xb yb →
  angle_APB_60 xp yp xa ya xb yb →
  (xp = 0 ∧ yp = 0) ∨ (xp = 8/5 ∧ yp = 4/5) :=
sorry

theorem fixed_points_on_circle :
  ∀ (xp yp xa ya : ℝ),
  P_on_line_l xp yp →
  tangents_to_M xp yp xa ya xp yp →
  ∃ (t : ℝ),
  (1 - t) * xp + t * xa = 0 ∧ (1 - t) * yp + t * ya = 2 ∨
  (1 - t) * xp + t * xa = 4/5 ∧ (1 - t) * yp + t * ya = 2/5 :=
sorry

end NUMINAMATH_CALUDE_tangent_points_coordinates_fixed_points_on_circle_l2168_216850


namespace NUMINAMATH_CALUDE_inequalities_always_hold_l2168_216830

theorem inequalities_always_hold :
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → c / a < c / b) ∧
  (∀ a b : ℝ, (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2) ∧
  (∀ a b : ℝ, a + b ≤ Real.sqrt (2 * (a^2 + b^2))) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_always_hold_l2168_216830


namespace NUMINAMATH_CALUDE_second_highest_coefficient_of_g_l2168_216863

/-- Given a polynomial g(x) satisfying g(x + 1) - g(x) = 6x^2 + 4x + 2 for all x,
    prove that the second highest coefficient of g(x) is 2/3 -/
theorem second_highest_coefficient_of_g (g : ℝ → ℝ) 
  (h : ∀ x, g (x + 1) - g x = 6 * x^2 + 4 * x + 2) :
  ∃ a b c d : ℝ, (∀ x, g x = a * x^3 + b * x^2 + c * x + d) ∧ b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_second_highest_coefficient_of_g_l2168_216863


namespace NUMINAMATH_CALUDE_plan_C_not_more_expensive_l2168_216837

/-- Represents the number of days required for Team A to complete the project alone -/
def x : ℕ := sorry

/-- Cost per day for Team A -/
def cost_A : ℕ := 10000

/-- Cost per day for Team B -/
def cost_B : ℕ := 6000

/-- Number of days both teams work together in Plan C -/
def days_together : ℕ := 3

/-- Extra days required for Team B to complete the project alone -/
def extra_days_B : ℕ := 4

/-- Equation representing the work done in Plan C -/
axiom plan_C_equation : (days_together : ℝ) / x + x / (x + extra_days_B) = 1

/-- Cost of Plan A -/
def cost_plan_A : ℕ := x * cost_A

/-- Cost of Plan C -/
def cost_plan_C : ℕ := days_together * (cost_A + cost_B) + (x - days_together) * cost_B

/-- Theorem stating that Plan C is not more expensive than Plan A -/
theorem plan_C_not_more_expensive : cost_plan_C ≤ cost_plan_A := by sorry

end NUMINAMATH_CALUDE_plan_C_not_more_expensive_l2168_216837


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2168_216876

/-- 
Given an arithmetic sequence with three terms where the first term is 3² and the third term is 3⁴,
prove that the second term (z) is equal to 45.
-/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence condition
    a 0 = 3^2 →                                       -- first term is 3²
    a 2 = 3^4 →                                       -- third term is 3⁴
    a 1 = 45 :=                                       -- second term (z) is 45
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2168_216876


namespace NUMINAMATH_CALUDE_inequality_theorem_l2168_216814

theorem inequality_theorem (n : ℕ) (hn : n > 0) :
  (∀ x : ℝ, x > 0 → x + (n^n : ℝ) / x^n ≥ n + 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → x + a / x^n ≥ n + 1) → a = n^n) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2168_216814


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2168_216860

/-- The total surface area of a rectangular solid -/
def total_surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 5 meters, width 4 meters, and depth 1 meter is 58 square meters -/
theorem rectangular_solid_surface_area :
  total_surface_area 5 4 1 = 58 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2168_216860


namespace NUMINAMATH_CALUDE_combined_transformation_correct_l2168_216886

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -2; 2, 0]

theorem combined_transformation_correct :
  combined_transformation = rotation_matrix_90_ccw * dilation_matrix 2 := by
  sorry

end NUMINAMATH_CALUDE_combined_transformation_correct_l2168_216886


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2168_216807

theorem rationalize_denominator :
  ∃ (A B C : ℕ) (D : ℕ+),
    (1 : ℝ) / (Real.rpow 5 (1/3) - Real.rpow 4 (1/3)) =
    (Real.rpow A (1/3) + Real.rpow B (1/3) + Real.rpow C (1/3)) / D ∧
    A = 25 ∧ B = 20 ∧ C = 16 ∧ D = 1 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2168_216807


namespace NUMINAMATH_CALUDE_average_sale_is_7500_l2168_216810

def monthly_sales : List ℕ := [7435, 7920, 7855, 8230, 7560, 6000]

def total_sales : ℕ := monthly_sales.sum

def num_months : ℕ := monthly_sales.length

def average_sale : ℚ := (total_sales : ℚ) / (num_months : ℚ)

theorem average_sale_is_7500 : average_sale = 7500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_7500_l2168_216810


namespace NUMINAMATH_CALUDE_system_solution_l2168_216884

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (2*x - Real.sqrt (x*y) - 4*Real.sqrt (x/y) + 2 = 0 ∧
   2*x^2 + x^2*y^4 = 18*y^2) →
  ((x = 2 ∧ y = 2) ∨
   (x = (Real.sqrt (Real.sqrt 286))/4 ∧ y = Real.sqrt (Real.sqrt 286))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2168_216884


namespace NUMINAMATH_CALUDE_sqrt_two_between_one_and_two_l2168_216821

theorem sqrt_two_between_one_and_two :
  1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_between_one_and_two_l2168_216821


namespace NUMINAMATH_CALUDE_max_leftover_oranges_l2168_216898

theorem max_leftover_oranges (n : ℕ) : ∃ (q : ℕ), n = 8 * q + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_leftover_oranges_l2168_216898


namespace NUMINAMATH_CALUDE_square_roots_of_nine_l2168_216875

theorem square_roots_of_nine :
  {x : ℝ | x ^ 2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_roots_of_nine_l2168_216875


namespace NUMINAMATH_CALUDE_tank_filling_l2168_216825

/-- Proves that adding 4 gallons to a 32-gallon tank that is 3/4 full results in the tank being 7/8 full -/
theorem tank_filling (tank_capacity : ℚ) (initial_fraction : ℚ) (added_amount : ℚ) : 
  tank_capacity = 32 →
  initial_fraction = 3 / 4 →
  added_amount = 4 →
  (initial_fraction * tank_capacity + added_amount) / tank_capacity = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_l2168_216825


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l2168_216871

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2, c = 2√3, and C = π/3, then b = 4 -/
theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  (a = 2) → (c = 2 * Real.sqrt 3) → (C = π / 3) → (b = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l2168_216871


namespace NUMINAMATH_CALUDE_four_digit_permutations_eq_six_l2168_216817

/-- The number of different positive, four-digit integers that can be formed using the digits 3, 3, 8, and 8 -/
def four_digit_permutations : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, four-digit integers
    that can be formed using the digits 3, 3, 8, and 8 is equal to 6 -/
theorem four_digit_permutations_eq_six :
  four_digit_permutations = 6 := by
  sorry

#eval four_digit_permutations

end NUMINAMATH_CALUDE_four_digit_permutations_eq_six_l2168_216817


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2168_216846

theorem simplify_square_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^4) = 225 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2168_216846


namespace NUMINAMATH_CALUDE_problem_solution_l2168_216882

theorem problem_solution (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 119) : 
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2168_216882


namespace NUMINAMATH_CALUDE_no_zero_points_implies_a_leq_two_l2168_216874

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1) - 2 * Real.log x

theorem no_zero_points_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0) →
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_no_zero_points_implies_a_leq_two_l2168_216874


namespace NUMINAMATH_CALUDE_distance_is_sqrt_6_l2168_216803

def A : ℝ × ℝ × ℝ := (1, -1, -1)
def P : ℝ × ℝ × ℝ := (1, 1, 1)
def direction_vector : ℝ × ℝ × ℝ := (1, 0, -1)

def distance_point_to_line (P : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_6 :
  distance_point_to_line P A direction_vector = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_6_l2168_216803


namespace NUMINAMATH_CALUDE_triangle_angle_from_side_ratio_l2168_216888

theorem triangle_angle_from_side_ratio :
  ∀ (a b c : ℝ) (A B C : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a / b = 1 / Real.sqrt 3) →
  (a / c = 1 / 2) →
  (A + B + C = π) →
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →
  (b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) →
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) →
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_from_side_ratio_l2168_216888


namespace NUMINAMATH_CALUDE_inequality_proof_l2168_216826

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2168_216826


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2168_216851

theorem binomial_expansion_coefficients :
  let n : ℕ := 50
  let a : ℕ := 2
  -- Coefficient of x^3
  (n.choose 3) * a^(n - 3) = 19600 * 2^47 ∧
  -- Constant term
  (n.choose 0) * a^n = 2^50 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2168_216851


namespace NUMINAMATH_CALUDE_nancy_balloon_count_l2168_216824

/-- Given that Mary has 7 balloons and Nancy has 4 times as many balloons as Mary,
    prove that Nancy has 28 balloons. -/
theorem nancy_balloon_count :
  ∀ (mary_balloons nancy_balloons : ℕ),
    mary_balloons = 7 →
    nancy_balloons = 4 * mary_balloons →
    nancy_balloons = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_nancy_balloon_count_l2168_216824


namespace NUMINAMATH_CALUDE_boat_speed_difference_l2168_216858

/-- Proves that the difference between boat speed and current speed in a channel is 1 km/h -/
theorem boat_speed_difference (V : ℝ) : ∃ (U : ℝ),
  (1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) ∧ (U - V = 1) :=
by
  sorry

#check boat_speed_difference

end NUMINAMATH_CALUDE_boat_speed_difference_l2168_216858


namespace NUMINAMATH_CALUDE_tina_shoe_expense_l2168_216833

def savings_june : ℕ := 27
def savings_july : ℕ := 14
def savings_august : ℕ := 21
def spent_on_books : ℕ := 5
def amount_left : ℕ := 40

theorem tina_shoe_expense : 
  savings_june + savings_july + savings_august - spent_on_books - amount_left = 17 := by
  sorry

end NUMINAMATH_CALUDE_tina_shoe_expense_l2168_216833


namespace NUMINAMATH_CALUDE_smallest_class_size_l2168_216866

theorem smallest_class_size : 
  (∃ n : ℕ, n > 30 ∧ 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
      n = 3 * x + y ∧ 
      y = x + 1) ∧
    (∀ m : ℕ, m > 30 → 
      (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
        m = 3 * a + b ∧ 
        b = a + 1) → 
      m ≥ n)) →
  (∃ n : ℕ, n = 33 ∧ n > 30 ∧ 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
      n = 3 * x + y ∧ 
      y = x + 1) ∧
    (∀ m : ℕ, m > 30 → 
      (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
        m = 3 * a + b ∧ 
        b = a + 1) → 
      m ≥ n)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2168_216866


namespace NUMINAMATH_CALUDE_sampling_probabilities_equal_l2168_216865

/-- Represents the composition of a batch of components -/
structure BatchComposition where
  total : ℕ
  first_class : ℕ
  second_class : ℕ
  third_class : ℕ
  unqualified : ℕ

/-- Represents the probabilities of selecting an individual component using different sampling methods -/
structure SamplingProbabilities where
  simple_random : ℚ
  stratified : ℚ
  systematic : ℚ

/-- Theorem stating that all sampling probabilities are equal to 1/8 for the given batch composition and sample size -/
theorem sampling_probabilities_equal (batch : BatchComposition) (sample_size : ℕ) 
  (h1 : batch.total = 160)
  (h2 : batch.first_class = 48)
  (h3 : batch.second_class = 64)
  (h4 : batch.third_class = 32)
  (h5 : batch.unqualified = 16)
  (h6 : sample_size = 20)
  (h7 : batch.total = batch.first_class + batch.second_class + batch.third_class + batch.unqualified) :
  ∃ (probs : SamplingProbabilities), 
    probs.simple_random = 1/8 ∧ 
    probs.stratified = 1/8 ∧ 
    probs.systematic = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_sampling_probabilities_equal_l2168_216865


namespace NUMINAMATH_CALUDE_inequality_relationship_l2168_216877

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def monotone_decreasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x < y → y ≤ 1 → f y < f x

theorem inequality_relationship (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : has_period_two f) 
  (h3 : monotone_decreasing_on_unit_interval f) : 
  f (-1) < f 2.5 ∧ f 2.5 < f 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l2168_216877


namespace NUMINAMATH_CALUDE_oranges_added_correct_l2168_216880

/-- The number of oranges added to make apples 50% of the total fruit -/
def oranges_added (initial_apples initial_oranges : ℕ) : ℕ :=
  let total := initial_apples + initial_oranges
  (2 * total) - initial_oranges

theorem oranges_added_correct (initial_apples initial_oranges : ℕ) :
  initial_apples = 10 →
  initial_oranges = 5 →
  oranges_added initial_apples initial_oranges = 5 :=
by
  sorry

#eval oranges_added 10 5

end NUMINAMATH_CALUDE_oranges_added_correct_l2168_216880


namespace NUMINAMATH_CALUDE_tetrahedron_intersection_theorem_l2168_216838

/-- Represents a tetrahedron with an inscribed sphere -/
structure TetrahedronWithSphere where
  volume : ℝ
  surface_area : ℝ
  inscribed_sphere_radius : ℝ

/-- Represents a plane intersecting three edges of a tetrahedron -/
structure IntersectingPlane where
  passes_through_center : Bool

/-- Represents the parts of the tetrahedron created by the intersecting plane -/
structure TetrahedronParts where
  volume_ratio : ℝ
  surface_area_ratio : ℝ

/-- The main theorem statement -/
theorem tetrahedron_intersection_theorem 
  (t : TetrahedronWithSphere) 
  (p : IntersectingPlane) 
  (parts : TetrahedronParts) : 
  (parts.volume_ratio = parts.surface_area_ratio) ↔ p.passes_through_center := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_intersection_theorem_l2168_216838


namespace NUMINAMATH_CALUDE_empty_intersection_l2168_216809

def S : Set ℚ := {x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℚ := x - 1/x

def f_iter (n : ℕ) : Set ℚ → Set ℚ :=
  match n with
  | 0 => id
  | n + 1 => f_iter n ∘ (λ s => f '' s)

theorem empty_intersection :
  (⋂ n : ℕ, f_iter n S) = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_intersection_l2168_216809


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2168_216815

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes x ± 2y = 0 is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let asymptote (x y : ℝ) := x = 2 * y ∨ x = -2 * y
  asymptote x y ∧ x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2168_216815


namespace NUMINAMATH_CALUDE_nelly_winning_bid_l2168_216867

-- Define Joe's bid
def joes_bid : ℕ := 160000

-- Define Nelly's bid calculation
def nellys_bid : ℕ := 3 * joes_bid + 2000

-- Theorem to prove
theorem nelly_winning_bid : nellys_bid = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_winning_bid_l2168_216867
