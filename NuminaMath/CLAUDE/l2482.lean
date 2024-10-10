import Mathlib

namespace ice_pop_price_l2482_248218

theorem ice_pop_price :
  ∀ (price : ℝ),
  (∃ (xiaoming_money xiaodong_money : ℝ),
    xiaoming_money = price - 0.5 ∧
    xiaodong_money = price - 1 ∧
    xiaoming_money + xiaodong_money < price) →
  price = 1 :=
by
  sorry

end ice_pop_price_l2482_248218


namespace not_much_different_from_2023_l2482_248231

theorem not_much_different_from_2023 (x : ℝ) : 
  (x - 2023 ≤ 0) ↔ (x ≤ 2023) :=
by sorry

end not_much_different_from_2023_l2482_248231


namespace annual_rent_per_square_foot_l2482_248283

/-- Calculates the annual rent per square foot for a shop -/
theorem annual_rent_per_square_foot
  (length : ℝ) (width : ℝ) (monthly_rent : ℝ)
  (h1 : length = 18)
  (h2 : width = 22)
  (h3 : monthly_rent = 2244) :
  (monthly_rent * 12) / (length * width) = 68 := by
  sorry

end annual_rent_per_square_foot_l2482_248283


namespace olga_aquarium_fish_count_l2482_248263

/-- The number of fish in Olga's aquarium -/
def fish_count : ℕ := 76

/-- The colors of fish in the aquarium -/
inductive FishColor
| Yellow | Blue | Green | Orange | Purple | Pink | Grey | Other

/-- The count of fish for each color -/
def fish_by_color (color : FishColor) : ℕ :=
  match color with
  | FishColor.Yellow => 12
  | FishColor.Blue => 6
  | FishColor.Green => 24
  | FishColor.Purple => 3
  | FishColor.Pink => 8
  | _ => 0  -- We don't have exact numbers for Orange, Grey, and Other

theorem olga_aquarium_fish_count :
  fish_count = 76 ∧
  fish_by_color FishColor.Yellow = 12 ∧
  fish_by_color FishColor.Blue = fish_by_color FishColor.Yellow / 2 ∧
  fish_by_color FishColor.Green = 2 * fish_by_color FishColor.Yellow ∧
  fish_by_color FishColor.Purple = fish_by_color FishColor.Blue / 2 ∧
  fish_by_color FishColor.Pink = fish_by_color FishColor.Green / 3 ∧
  (fish_count : ℚ) = (fish_by_color FishColor.Yellow +
                      fish_by_color FishColor.Blue +
                      fish_by_color FishColor.Green +
                      fish_by_color FishColor.Purple +
                      fish_by_color FishColor.Pink) / 0.7 :=
by sorry

#check olga_aquarium_fish_count

end olga_aquarium_fish_count_l2482_248263


namespace hyperbola_from_ellipse_l2482_248274

/-- Given an ellipse with equation x²/24 + y²/49 = 1, 
    prove that the equation of the hyperbola whose vertices are the foci of this ellipse 
    and whose foci are the vertices of this ellipse is y²/25 - x²/24 = 1 -/
theorem hyperbola_from_ellipse (x y : ℝ) :
  (x^2 / 24 + y^2 / 49 = 1) →
  ∃ (x' y' : ℝ), (y'^2 / 25 - x'^2 / 24 = 1 ∧ 
    (∀ (a b c : ℝ), (a^2 = 49 ∧ b^2 = 24 ∧ c^2 = a^2 - b^2) →
      ((x' = 0 ∧ y' = c) ∨ (x' = 0 ∧ y' = -c)) ∧
      ((x' = 0 ∧ y' = a) ∨ (x' = 0 ∧ y' = -a)))) :=
by sorry


end hyperbola_from_ellipse_l2482_248274


namespace twenty_seven_binary_l2482_248240

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number in base 2 -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem twenty_seven_binary :
  toBinary 27 = [true, true, false, true, true] :=
sorry

end twenty_seven_binary_l2482_248240


namespace sphere_surface_area_l2482_248216

/-- The surface area of a sphere, given properties of a hemisphere. -/
theorem sphere_surface_area (r : ℝ) (h_base_area : π * r^2 = 3) (h_hemisphere_area : 3 * π * r^2 = 9) :
  ∃ A : ℝ → ℝ, A r = 4 * π * r^2 := by sorry

end sphere_surface_area_l2482_248216


namespace product_of_special_integers_l2482_248288

theorem product_of_special_integers (A B C D : ℕ+) 
  (sum_eq : A + B + C + D = 70)
  (def_A : A = 3 * C + 1)
  (def_B : B = 3 * C + 5)
  (def_D : D = 3 * C * C) :
  A * B * C * D = 16896 := by
  sorry

end product_of_special_integers_l2482_248288


namespace stamp_cost_l2482_248234

theorem stamp_cost (total_cost : ℕ) (num_stamps : ℕ) (h1 : total_cost = 136) (h2 : num_stamps = 4) :
  total_cost / num_stamps = 34 := by
  sorry

end stamp_cost_l2482_248234


namespace square_greater_than_self_for_x_greater_than_one_l2482_248265

theorem square_greater_than_self_for_x_greater_than_one (x : ℝ) : x > 1 → x^2 > x := by
  sorry

end square_greater_than_self_for_x_greater_than_one_l2482_248265


namespace height_difference_l2482_248201

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := h.feet * 12 + h.inches

/-- Mark's height -/
def markHeight : Height := ⟨5, 3⟩

/-- Mike's height -/
def mikeHeight : Height := ⟨6, 1⟩

theorem height_difference : heightToInches mikeHeight - heightToInches markHeight = 10 := by
  sorry

end height_difference_l2482_248201


namespace sibling_difference_l2482_248277

/-- Given the number of siblings for Masud, calculate the number of siblings for Janet -/
def janet_siblings (masud_siblings : ℕ) : ℕ :=
  4 * masud_siblings - 60

/-- Given the number of siblings for Masud, calculate the number of siblings for Carlos -/
def carlos_siblings (masud_siblings : ℕ) : ℕ :=
  (3 * masud_siblings) / 4

/-- Theorem stating the difference in siblings between Janet and Carlos -/
theorem sibling_difference (masud_siblings : ℕ) (h : masud_siblings = 60) :
  janet_siblings masud_siblings - carlos_siblings masud_siblings = 135 := by
  sorry


end sibling_difference_l2482_248277


namespace village_foods_tomato_sales_l2482_248241

theorem village_foods_tomato_sales (customers : ℕ) (lettuce_per_customer : ℕ) 
  (lettuce_price : ℚ) (tomato_price : ℚ) (total_sales : ℚ) 
  (h1 : customers = 500)
  (h2 : lettuce_per_customer = 2)
  (h3 : lettuce_price = 1)
  (h4 : tomato_price = 1/2)
  (h5 : total_sales = 2000) :
  (total_sales - (↑customers * ↑lettuce_per_customer * lettuce_price)) / (↑customers * tomato_price) = 4 := by
sorry

end village_foods_tomato_sales_l2482_248241


namespace correct_balloons_given_to_fred_l2482_248229

/-- The number of balloons Sam gave to Fred -/
def balloons_given_to_fred (sam_initial : ℕ) (mary : ℕ) (total : ℕ) : ℕ :=
  sam_initial - (total - mary)

theorem correct_balloons_given_to_fred :
  balloons_given_to_fred 6 7 8 = 5 := by
  sorry

end correct_balloons_given_to_fred_l2482_248229


namespace twenty_percent_greater_than_88_l2482_248299

theorem twenty_percent_greater_than_88 (x : ℝ) : x = 88 * 1.2 → x = 105.6 := by
  sorry

end twenty_percent_greater_than_88_l2482_248299


namespace marks_trip_length_l2482_248208

theorem marks_trip_length (total : ℚ) 
  (h1 : total / 4 + 30 + total / 6 = total) : 
  total = 360 / 7 := by
sorry

end marks_trip_length_l2482_248208


namespace vovochka_max_candies_l2482_248298

/-- Represents the problem of distributing candies among classmates. -/
structure CandyDistribution where
  totalCandies : Nat
  totalClassmates : Nat
  minGroupSize : Nat
  minGroupCandies : Nat

/-- Calculates the maximum number of candies Vovochka can keep. -/
def maxCandiesForVovochka (dist : CandyDistribution) : Nat :=
  sorry

/-- Theorem stating the maximum number of candies Vovochka can keep. -/
theorem vovochka_max_candies :
  let dist : CandyDistribution := {
    totalCandies := 200,
    totalClassmates := 25,
    minGroupSize := 16,
    minGroupCandies := 100
  }
  maxCandiesForVovochka dist = 37 := by sorry

end vovochka_max_candies_l2482_248298


namespace system_solution_correct_l2482_248251

theorem system_solution_correct (x y : ℝ) : 
  (x = 2 ∧ y = -2) → (x + 2*y = -2 ∧ 2*x + y = 2) := by
sorry

end system_solution_correct_l2482_248251


namespace bridge_length_l2482_248225

/-- The length of a bridge given specific train crossing conditions -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 100 →
  crossing_time = 36 →
  train_speed = 40 →
  train_speed * crossing_time - train_length = 1340 := by
sorry

end bridge_length_l2482_248225


namespace projection_result_l2482_248272

def a : ℝ × ℝ := (-4, 2)
def b : ℝ × ℝ := (3, 4)

theorem projection_result (v : ℝ × ℝ) (p : ℝ × ℝ) 
  (h1 : v ≠ (0, 0)) 
  (h2 : p = (v.1 * (a.1 * v.1 + a.2 * v.2) / (v.1^2 + v.2^2), 
             v.2 * (a.1 * v.1 + a.2 * v.2) / (v.1^2 + v.2^2)))
  (h3 : p = (v.1 * (b.1 * v.1 + b.2 * v.2) / (v.1^2 + v.2^2), 
             v.2 * (b.1 * v.1 + b.2 * v.2) / (v.1^2 + v.2^2))) :
  p = (-44/53, 154/53) := by
sorry

end projection_result_l2482_248272


namespace rectangle_width_l2482_248200

/-- Given a square and a rectangle, if the area of the square is five times the area of the rectangle,
    the perimeter of the square is 800 cm, and the length of the rectangle is 125 cm,
    then the width of the rectangle is 64 cm. -/
theorem rectangle_width (square_perimeter : ℝ) (rectangle_length : ℝ) :
  square_perimeter = 800 ∧
  rectangle_length = 125 ∧
  (square_perimeter / 4) ^ 2 = 5 * (rectangle_length * (64 : ℝ)) →
  64 = (square_perimeter / 4) ^ 2 / (5 * rectangle_length) :=
by sorry

end rectangle_width_l2482_248200


namespace product_5832_sum_62_l2482_248282

theorem product_5832_sum_62 : ∃ (a b c : ℕ+),
  (a.val > 1) ∧ (b.val > 1) ∧ (c.val > 1) ∧
  (a * b * c = 5832) ∧
  (Nat.gcd a.val b.val = 1) ∧ (Nat.gcd b.val c.val = 1) ∧ (Nat.gcd c.val a.val = 1) ∧
  (a + b + c = 62) := by
sorry

end product_5832_sum_62_l2482_248282


namespace symmetry_implies_values_l2482_248269

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_values (m n : ℝ) :
  symmetric_wrt_y_axis (-m, 3) (-5, n) → m = -5 ∧ n = 3 := by
  sorry

end symmetry_implies_values_l2482_248269


namespace total_chocolate_bars_l2482_248237

/-- The number of chocolate bars in a large box -/
def chocolate_bars_in_large_box (small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  small_boxes * bars_per_small_box

/-- Theorem: There are 640 chocolate bars in the large box -/
theorem total_chocolate_bars :
  chocolate_bars_in_large_box 20 32 = 640 := by
sorry

end total_chocolate_bars_l2482_248237


namespace equal_area_segment_property_l2482_248260

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  midline_ratio : (b + 75) / (b + 150) = 3 / 4  -- Area ratio condition for midline
  h_pos : h > 0
  b_pos : b > 0

/-- The length of the segment that divides the trapezoid into two equal areas -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  225  -- This is the value of x we found in the solution

/-- The theorem to be proved -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(equal_area_segment t)^2 / 100⌋ = 506 := by
  sorry

end equal_area_segment_property_l2482_248260


namespace problem_1_l2482_248243

theorem problem_1 : -9 + 5 - (-12) + (-3) = 8 := by sorry

end problem_1_l2482_248243


namespace unique_number_pair_l2482_248246

theorem unique_number_pair : ∃! (x y : ℕ), 
  100 ≤ x ∧ x < 1000 ∧ 
  1000 ≤ y ∧ y < 10000 ∧ 
  10000 * x + y = 12 * x * y ∧
  x + y = 1083 := by
  sorry

end unique_number_pair_l2482_248246


namespace rectangular_box_volume_l2482_248217

theorem rectangular_box_volume 
  (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := by
sorry

end rectangular_box_volume_l2482_248217


namespace sum_squares_s_r_l2482_248221

def r : Finset Int := {-2, -1, 0, 1, 3}
def r_range : Finset Int := {-1, 0, 3, 4, 6}

def s_domain : Finset Int := {0, 1, 2, 3, 4, 5}
def s (x : Int) : Int := x^2 + x + 1

theorem sum_squares_s_r : 
  (r_range ∩ s_domain).sum (fun x => (s x)^2) = 611 := by
  sorry

end sum_squares_s_r_l2482_248221


namespace evaluate_expression_l2482_248204

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -12) :
  x^2 * y^3 * z = -1/36 := by
  sorry

end evaluate_expression_l2482_248204


namespace factoring_theorem_l2482_248206

theorem factoring_theorem (x : ℝ) : x^2 * (x + 3) + 2 * (x + 3) + (x + 3) = (x^2 + 3) * (x + 3) := by
  sorry

end factoring_theorem_l2482_248206


namespace complex_number_properties_l2482_248226

theorem complex_number_properties : 
  (∃ (s₁ s₂ : Prop) (s₃ s₄ : Prop), 
    s₁ ∧ s₂ ∧ ¬s₃ ∧ ¬s₄ ∧
    s₁ = (∀ z₁ z₂ : ℂ, z₁ * z₂ = z₂ * z₁) ∧
    s₂ = (∀ z₁ z₂ : ℂ, Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂) ∧
    s₃ = (∀ z : ℂ, Complex.abs z = 1 → z = 1 ∨ z = -1) ∧
    s₄ = (∀ z : ℂ, (Complex.abs z)^2 = z^2)) :=
by sorry

end complex_number_properties_l2482_248226


namespace rotate_W_180_is_M_l2482_248203

/-- Represents an uppercase English letter -/
inductive UppercaseLetter
| W
| M

/-- Represents a geometric figure -/
class GeometricFigure where
  /-- Indicates if the figure is axisymmetric -/
  is_axisymmetric : Bool

/-- Represents the result of rotating a letter -/
def rotate_180_degrees (letter : UppercaseLetter) (is_axisymmetric : Bool) : UppercaseLetter :=
  sorry

/-- Theorem: Rotating W 180° results in M -/
theorem rotate_W_180_is_M :
  ∀ (w : UppercaseLetter) (fig : GeometricFigure),
    w = UppercaseLetter.W →
    fig.is_axisymmetric = true →
    rotate_180_degrees w fig.is_axisymmetric = UppercaseLetter.M :=
  sorry

end rotate_W_180_is_M_l2482_248203


namespace simple_random_sampling_probability_l2482_248215

theorem simple_random_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 100) 
  (h2 : sample_size = 30) :
  (sample_size : ℚ) / (population_size : ℚ) = 3 / 10 := by
  sorry

end simple_random_sampling_probability_l2482_248215


namespace blue_pens_removed_l2482_248207

/-- Represents the number of pens in a jar -/
structure JarContents where
  blue : ℕ
  black : ℕ
  red : ℕ

/-- The initial contents of the jar -/
def initial_jar : JarContents := ⟨9, 21, 6⟩

/-- The number of black pens removed -/
def black_pens_removed : ℕ := 7

/-- The final number of pens in the jar after removals -/
def final_pens : ℕ := 25

/-- Theorem stating that 4 blue pens were removed -/
theorem blue_pens_removed :
  ∃ (x : ℕ),
    x = 4 ∧
    initial_jar.blue - x +
    (initial_jar.black - black_pens_removed) +
    initial_jar.red = final_pens :=
  sorry

end blue_pens_removed_l2482_248207


namespace area_ratio_eab_abcd_l2482_248255

/-- Represents a trapezoid ABCD with an extended triangle EAB -/
structure ExtendedTrapezoid where
  /-- Length of base AB -/
  ab : ℝ
  /-- Length of base CD -/
  cd : ℝ
  /-- Height of the trapezoid -/
  h : ℝ
  /-- Assertion that AB = 7 -/
  ab_eq : ab = 7
  /-- Assertion that CD = 15 -/
  cd_eq : cd = 15
  /-- Assertion that the height of triangle EAB is thrice the height of the trapezoid -/
  eab_height : ℝ
  eab_height_eq : eab_height = 3 * h

/-- The ratio of the area of triangle EAB to the area of trapezoid ABCD is 21/22 -/
theorem area_ratio_eab_abcd (t : ExtendedTrapezoid) : 
  (t.ab * t.eab_height) / ((t.ab + t.cd) * t.h) = 21 / 22 := by
  sorry

end area_ratio_eab_abcd_l2482_248255


namespace rectangle_in_circle_distances_l2482_248253

theorem rectangle_in_circle_distances (a b : ℝ) (ha : a = 24) (hb : b = 7) :
  let r := (a^2 + b^2).sqrt / 2
  let of := ((r^2 - (a/2)^2).sqrt : ℝ)
  let mf := r - of
  let mk := r + of
  ((mf^2 + (a/2)^2).sqrt, (mk^2 + (a/2)^2).sqrt) = (15, 20) := by
  sorry

end rectangle_in_circle_distances_l2482_248253


namespace solve_custom_equation_l2482_248235

-- Define the custom operation
def custom_op (m n : ℤ) : ℤ := n^2 - m

-- Theorem statement
theorem solve_custom_equation :
  ∀ x : ℤ, custom_op x 3 = 5 → x = 4 := by
  sorry

end solve_custom_equation_l2482_248235


namespace a_plus_b_value_l2482_248209

theorem a_plus_b_value (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end a_plus_b_value_l2482_248209


namespace polygon_sides_from_angle_sum_l2482_248254

/-- The number of sides of a polygon given the sum of its interior angles -/
theorem polygon_sides_from_angle_sum (sum_of_angles : ℝ) : sum_of_angles = 720 → ∃ n : ℕ, n = 6 ∧ (n - 2) * 180 = sum_of_angles := by
  sorry

end polygon_sides_from_angle_sum_l2482_248254


namespace solution_range_l2482_248257

theorem solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, x^2 + 2*x - a = 0) ↔ a ∈ Set.Icc (-1) 3 := by
  sorry

end solution_range_l2482_248257


namespace line_perp_plane_implies_planes_perp_l2482_248280

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : subset m α)
  (h4 : perp m β) :
  perp_planes α β :=
sorry

end line_perp_plane_implies_planes_perp_l2482_248280


namespace cube_sum_reciprocal_l2482_248286

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end cube_sum_reciprocal_l2482_248286


namespace truth_and_lie_l2482_248270

/-- Represents a person who either always tells the truth or always lies -/
inductive Person
| Truthful
| Liar

/-- The setup of three people sitting side by side -/
structure Setup :=
  (left : Person)
  (middle : Person)
  (right : Person)

/-- The statement made by the left person about the middle person's response -/
def leftStatement (s : Setup) : Prop :=
  s.middle = Person.Truthful

/-- The statement made by the right person about the middle person's response -/
def rightStatement (s : Setup) : Prop :=
  s.middle = Person.Liar

theorem truth_and_lie (s : Setup) :
  (leftStatement s = true ↔ s.left = Person.Truthful) ∧
  (rightStatement s = false ↔ s.right = Person.Liar) :=
sorry

end truth_and_lie_l2482_248270


namespace munchausen_polygon_theorem_l2482_248247

/-- A polygon in a 2D plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.length ≥ 3

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- A line in a 2D plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a polygon -/
def is_inside (p : Point) (poly : Polygon) : Prop := sorry

/-- Counts the number of regions a line divides a polygon into -/
def count_regions (l : Line) (poly : Polygon) : ℕ := sorry

/-- Theorem: There exists a polygon and a point inside it such that 
    any line passing through this point divides the polygon into 
    exactly three smaller polygons -/
theorem munchausen_polygon_theorem : 
  ∃ (poly : Polygon) (p : Point), 
    is_inside p poly ∧ 
    ∀ (l : Line), l.point1 = p ∨ l.point2 = p → count_regions l poly = 3 := by
  sorry

end munchausen_polygon_theorem_l2482_248247


namespace straight_line_shortest_l2482_248213

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line segment between two points
def LineSegment (p1 p2 : Point2D) : Set Point2D :=
  {p : Point2D | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y)}

-- Define the length of a path between two points
def PathLength (path : Set Point2D) : ℝ := sorry

-- Theorem: The straight line segment between two points has the shortest length among all paths between those points
theorem straight_line_shortest (p1 p2 : Point2D) :
  ∀ path : Set Point2D, p1 ∈ path ∧ p2 ∈ path →
    PathLength (LineSegment p1 p2) ≤ PathLength path :=
sorry

end straight_line_shortest_l2482_248213


namespace prime_sum_product_93_178_l2482_248279

theorem prime_sum_product_93_178 : 
  ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ p + q = 93 ∧ p * q = 178 := by
  sorry

end prime_sum_product_93_178_l2482_248279


namespace circle_to_bar_graph_correspondence_l2482_248296

/-- Represents the proportions of a circle graph -/
structure CircleGraph where
  white : ℝ
  black : ℝ
  gray : ℝ
  sum_to_one : white + black + gray = 1
  white_twice_others : white = 2 * black ∧ white = 2 * gray
  black_gray_equal : black = gray

/-- Represents the heights of bars in a bar graph -/
structure BarGraph where
  white : ℝ
  black : ℝ
  gray : ℝ

/-- Theorem stating that a bar graph correctly represents a circle graph -/
theorem circle_to_bar_graph_correspondence (cg : CircleGraph) (bg : BarGraph) :
  (bg.white = 2 * bg.black ∧ bg.white = 2 * bg.gray) ∧ bg.black = bg.gray :=
by sorry

end circle_to_bar_graph_correspondence_l2482_248296


namespace f_inequality_l2482_248293

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + a

-- State the theorem
theorem f_inequality (a b : ℝ) :
  (f a b 0 = 3) →
  (∀ x, f a b (2 - x) = f a b x) →
  (∀ x, f a b (b^x) ≤ f a b (a^x)) :=
by sorry

end f_inequality_l2482_248293


namespace decimal_sum_equals_fraction_l2482_248291

theorem decimal_sum_equals_fraction : 
  0.2 + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end decimal_sum_equals_fraction_l2482_248291


namespace unique_divisibility_pair_l2482_248297

/-- A predicate that checks if there are infinitely many positive integers k 
    for which (k^n + k^2 - 1) divides (k^m + k - 1) -/
def InfinitelyManyDivisors (m n : ℕ) : Prop :=
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ (k^n + k^2 - 1) ∣ (k^m + k - 1)

/-- The theorem stating that (5,3) is the only pair of integers (m,n) 
    satisfying the given conditions -/
theorem unique_divisibility_pair :
  ∀ m n : ℕ, m > 2 → n > 2 → InfinitelyManyDivisors m n → m = 5 ∧ n = 3 :=
sorry

end unique_divisibility_pair_l2482_248297


namespace linear_function_quadrants_l2482_248220

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a quadrant in the Cartesian plane -/
inductive Quadrant
  | I   -- x > 0, y > 0
  | II  -- x < 0, y > 0
  | III -- x < 0, y < 0
  | IV  -- x > 0, y < 0

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I   => ∃ x > 0, f.m * x + f.b > 0
  | Quadrant.II  => ∃ x < 0, f.m * x + f.b > 0
  | Quadrant.III => ∃ x < 0, f.m * x + f.b < 0
  | Quadrant.IV  => ∃ x > 0, f.m * x + f.b < 0

/-- The main theorem to prove -/
theorem linear_function_quadrants (f : LinearFunction) 
  (h1 : f.m = 4) 
  (h2 : f.b = 2) : 
  passesThrough f Quadrant.I ∧ 
  passesThrough f Quadrant.II ∧ 
  passesThrough f Quadrant.III :=
sorry

end linear_function_quadrants_l2482_248220


namespace tennis_ball_ratio_l2482_248295

/-- Given the number of tennis balls for Lily, Brian, and Frodo, prove the ratio of Brian's to Frodo's tennis balls -/
theorem tennis_ball_ratio :
  ∀ (lily_balls brian_balls frodo_balls : ℕ),
    lily_balls = 3 →
    brian_balls = 22 →
    frodo_balls = lily_balls + 8 →
    (brian_balls : ℚ) / (frodo_balls : ℚ) = 2 / 1 := by
  sorry

end tennis_ball_ratio_l2482_248295


namespace f_monotone_increasing_local_max_condition_l2482_248245

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (x - 1) + Real.log x - (a + 1) * x

theorem f_monotone_increasing (x : ℝ) (hx : x > 0) :
  let f₁ := f 1
  (deriv f₁) x ≥ 0 := by sorry

theorem local_max_condition (a : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a x ≤ f a 1) ↔ a < 1 := by sorry

end f_monotone_increasing_local_max_condition_l2482_248245


namespace complex_equation_solution_l2482_248275

theorem complex_equation_solution (z : ℂ) : (Complex.I * (z - 1) = 1 - Complex.I) → z = -Complex.I := by
  sorry

end complex_equation_solution_l2482_248275


namespace min_value_problem_l2482_248242

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 3/b = 1 → 2*x + 3*y ≤ 2*a + 3*b ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ 2/c + 3/d = 1 ∧ 2*c + 3*d = 25 :=
sorry

end min_value_problem_l2482_248242


namespace third_grade_boys_count_l2482_248228

/-- The number of third-grade boys in an elementary school -/
def third_grade_boys (total : ℕ) (fourth_grade_excess : ℕ) (third_grade_girl_deficit : ℕ) : ℕ :=
  let third_graders := (total - fourth_grade_excess) / 2
  let third_grade_boys := (third_graders + third_grade_girl_deficit) / 2
  third_grade_boys

/-- Theorem stating the number of third-grade boys given the conditions -/
theorem third_grade_boys_count :
  third_grade_boys 531 31 22 = 136 :=
by sorry

end third_grade_boys_count_l2482_248228


namespace jacket_savings_percentage_l2482_248285

/-- Calculates the percentage saved on a purchase given the original price and total savings. -/
def percentage_saved (original_price savings : ℚ) : ℚ :=
  (savings / original_price) * 100

/-- Proves that the total percentage saved on a jacket purchase is 22.5% given the specified conditions. -/
theorem jacket_savings_percentage :
  let original_price : ℚ := 160
  let store_discount : ℚ := 20
  let coupon_savings : ℚ := 16
  let total_savings : ℚ := store_discount + coupon_savings
  percentage_saved original_price total_savings = 22.5 := by
  sorry


end jacket_savings_percentage_l2482_248285


namespace arithmetic_sequence_ratio_l2482_248224

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_ratio (a₁ d : ℝ) :
  a₁ ≠ 0 →
  d ≠ 0 →
  (arithmetic_sequence a₁ d 2) * (arithmetic_sequence a₁ d 8) = (arithmetic_sequence a₁ d 4)^2 →
  (arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 6 + arithmetic_sequence a₁ d 9) /
  (arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 5) = 2 := by
sorry

end arithmetic_sequence_ratio_l2482_248224


namespace mass_of_man_on_boat_l2482_248239

/-- The mass of a man who causes a boat to sink by a certain depth -/
def mass_of_man (length breadth depth_sunk : ℝ) (water_density : ℝ) : ℝ :=
  length * breadth * depth_sunk * water_density

/-- Theorem stating that the mass of the man is 60 kg -/
theorem mass_of_man_on_boat :
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

end mass_of_man_on_boat_l2482_248239


namespace principal_amount_proof_l2482_248233

/-- 
Given a principal amount P put at simple interest for 3 years,
if increasing the interest rate by 3% results in 81 more interest,
then P must equal 900.
-/
theorem principal_amount_proof (P : ℝ) (R : ℝ) : 
  (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81 → P = 900 := by
  sorry

end principal_amount_proof_l2482_248233


namespace complex_magnitude_squared_l2482_248290

theorem complex_magnitude_squared (z : ℂ) (h : 2 * z + Complex.abs z = -3 + 12 * Complex.I) : Complex.normSq z = 61 := by
  sorry

end complex_magnitude_squared_l2482_248290


namespace library_repacking_l2482_248261

theorem library_repacking (initial_packages : ℕ) (pamphlets_per_initial_package : ℕ) (pamphlets_per_new_package : ℕ) : 
  initial_packages = 1450 →
  pamphlets_per_initial_package = 42 →
  pamphlets_per_new_package = 45 →
  (initial_packages * pamphlets_per_initial_package) % pamphlets_per_new_package = 15 :=
by
  sorry

#check library_repacking

end library_repacking_l2482_248261


namespace quadratic_inequality_l2482_248202

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) :
  (∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 4) →
  f a b c 5 < f a b c (-1) ∧ f a b c (-1) < f a b c 2 := by
sorry

end quadratic_inequality_l2482_248202


namespace find_number_l2482_248294

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 75 := by
  sorry

end find_number_l2482_248294


namespace average_annual_growth_rate_l2482_248273

/-- Given growth rates a and b for two consecutive years, 
    the average annual growth rate over these two years 
    is equal to √((a+1)(b+1)) - 1 -/
theorem average_annual_growth_rate 
  (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a + 1) * (b + 1)) - 1 ∧ 
  (1 + x)^2 = (1 + a) * (1 + b) :=
by sorry

end average_annual_growth_rate_l2482_248273


namespace circle_area_sum_l2482_248244

/-- The sum of the areas of all circles in an infinite sequence, where the radii form a geometric
    sequence with first term 10/3 and common ratio 4/9, is equal to 180π/13. -/
theorem circle_area_sum : 
  let r₁ : ℝ := 10 / 3  -- First term of the radii sequence
  let r : ℝ := 4 / 9    -- Common ratio of the radii sequence
  let area_sum := ∑' n, π * (r₁ * r ^ n) ^ 2  -- Sum of areas of all circles
  area_sum = 180 * π / 13 := by
  sorry

end circle_area_sum_l2482_248244


namespace zero_integer_not_positive_negative_l2482_248289

theorem zero_integer_not_positive_negative :
  (0 : ℤ) ∈ Set.univ ∧ (0 : ℤ) ∉ {x : ℤ | x > 0} ∧ (0 : ℤ) ∉ {x : ℤ | x < 0} := by
  sorry

end zero_integer_not_positive_negative_l2482_248289


namespace point_not_on_line_l2482_248227

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : 
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = a * x + c) :=
by sorry

end point_not_on_line_l2482_248227


namespace least_sum_problem_l2482_248223

theorem least_sum_problem (x y z : ℕ+) 
  (h1 : 4 * x.val = 6 * z.val)
  (h2 : ∀ (a b c : ℕ+), 4 * a.val = 6 * c.val → a.val + b.val + c.val ≥ 37)
  (h3 : x.val + y.val + z.val = 37) :
  y = 32 := by
sorry

end least_sum_problem_l2482_248223


namespace degree_to_radian_15_l2482_248230

theorem degree_to_radian_15 : 
  (15 : ℝ) * (π / 180) = π / 12 := by sorry

end degree_to_radian_15_l2482_248230


namespace rice_containers_l2482_248249

theorem rice_containers (total_weight : ℚ) (container_weight : ℕ) (pound_to_ounce : ℕ) :
  total_weight = 35 / 2 →
  container_weight = 70 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce : ℚ) / container_weight = 4 :=
by sorry

end rice_containers_l2482_248249


namespace a_1_greater_than_500_l2482_248219

theorem a_1_greater_than_500 (a : Fin 10000 → ℕ)
  (h1 : ∀ i j, i < j → a i < a j)
  (h2 : a 0 > 0)
  (h3 : a 9999 < 20000)
  (h4 : ∀ i j, i < j → Nat.gcd (a i) (a j) < a i) :
  500 < a 0 := by
  sorry

end a_1_greater_than_500_l2482_248219


namespace expression_value_l2482_248214

theorem expression_value : 
  let a : ℕ := 2017
  let b : ℕ := 2016
  let c : ℕ := 2015
  ((a^2 + b^2)^2 - c^2 - 4*a^2*b^2) / (a^2 + c - b^2) = 2018 := by
  sorry

end expression_value_l2482_248214


namespace max_value_expression_l2482_248264

theorem max_value_expression (x : ℝ) :
  (Real.exp (2 * x) + Real.exp (-2 * x) + 1) / (Real.exp x + Real.exp (-x) + 2) ≤ 2 * (1 - Real.sqrt 3) :=
sorry

end max_value_expression_l2482_248264


namespace negation_of_universal_proposition_l2482_248284

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l2482_248284


namespace prob_two_non_defective_pens_l2482_248212

/-- The probability of selecting 2 non-defective pens from a box of 9 pens with 3 defective pens -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h_total : total_pens = 9) 
  (h_defective : defective_pens = 3) : 
  (Nat.choose (total_pens - defective_pens) 2 : ℚ) / (Nat.choose total_pens 2) = 5 / 12 := by
  sorry

end prob_two_non_defective_pens_l2482_248212


namespace four_of_a_kind_count_l2482_248232

/-- Represents a standard deck of 52 playing cards --/
def Deck : Type := Fin 52

/-- Represents a 5-card hand --/
def Hand : Type := Finset Deck

/-- Returns true if a hand contains exactly four cards of the same value --/
def hasFourOfAKind (h : Hand) : Prop := sorry

/-- The number of 5-card hands containing exactly four cards of the same value --/
def numHandsWithFourOfAKind : ℕ := sorry

theorem four_of_a_kind_count : numHandsWithFourOfAKind = 624 := by sorry

end four_of_a_kind_count_l2482_248232


namespace factory_non_defective_percentage_l2482_248259

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : Real
  defective_rate : Real

/-- Calculates the percentage of non-defective products given a list of machines -/
def non_defective_percentage (machines : List Machine) : Real :=
  100 - (machines.map (λ m => m.production_percentage * m.defective_rate)).sum

/-- The theorem stating that the percentage of non-defective products is 95.25% -/
theorem factory_non_defective_percentage : 
  let machines : List Machine := [
    ⟨20, 2⟩,
    ⟨25, 4⟩,
    ⟨30, 5⟩,
    ⟨15, 7⟩,
    ⟨10, 8⟩
  ]
  non_defective_percentage machines = 95.25 := by
  sorry

end factory_non_defective_percentage_l2482_248259


namespace namjoon_walk_proof_l2482_248271

/-- The additional distance Namjoon walked compared to his usual route -/
def additional_distance (usual_distance initial_walk : ℝ) : ℝ :=
  2 * initial_walk + usual_distance - usual_distance

theorem namjoon_walk_proof (usual_distance initial_walk : ℝ) 
  (h1 : usual_distance = 1.2)
  (h2 : initial_walk = 0.3) :
  additional_distance usual_distance initial_walk = 0.6 := by
  sorry

#eval additional_distance 1.2 0.3

end namjoon_walk_proof_l2482_248271


namespace certain_number_multiplication_l2482_248292

theorem certain_number_multiplication (x : ℚ) : x / 11 = 2 → 6 * x = 132 := by
  sorry

end certain_number_multiplication_l2482_248292


namespace theatre_fraction_l2482_248268

/-- Represents the fraction of students in a school -/
structure SchoolFractions where
  pe : ℚ  -- Fraction of students who took P.E.
  theatre : ℚ  -- Fraction of students who took theatre
  music : ℚ  -- Fraction of students who took music

/-- Represents the fraction of students who left the school -/
structure LeavingFractions where
  pe : ℚ  -- Fraction of P.E. students who left
  theatre : ℚ  -- Fraction of theatre students who left

theorem theatre_fraction (s : SchoolFractions) (l : LeavingFractions) : 
  s.pe = 1/2 ∧ 
  s.pe + s.theatre + s.music = 1 ∧
  l.pe = 1/3 ∧
  l.theatre = 1/4 ∧
  (s.pe * (1 - l.pe) + s.music) / (1 - s.pe * l.pe - s.theatre * l.theatre) = 2/3 →
  s.theatre = 1/6 := by
  sorry

end theatre_fraction_l2482_248268


namespace birthday_number_proof_l2482_248287

theorem birthday_number_proof : ∃! T : ℕ+,
  (∃ x y : ℕ, 4 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
    T ^ 2 = 40000 + x * 1000 + y * 100 + 29) ∧
  T = 223 := by
  sorry

end birthday_number_proof_l2482_248287


namespace g_max_value_l2482_248276

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The maximum value of g(x) on the interval [0, 2] is 3 -/
theorem g_max_value : ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧ g c = 3 := by
  sorry

end g_max_value_l2482_248276


namespace percentage_of_120_to_50_l2482_248262

theorem percentage_of_120_to_50 : (120 : ℝ) / 50 * 100 = 240 := by
  sorry

end percentage_of_120_to_50_l2482_248262


namespace parabola_intersection_range_l2482_248278

/-- Given a line y = a intersecting the parabola y = x^2 at points A and B, 
    and a point C on the parabola such that angle ACB is a right angle, 
    the range of possible values for a is [1, +∞) -/
theorem parabola_intersection_range (a : ℝ) : 
  (∃ A B C : ℝ × ℝ, 
    (A.2 = a ∧ A.2 = A.1^2) ∧ 
    (B.2 = a ∧ B.2 = B.1^2) ∧ 
    (C.2 = C.1^2) ∧ 
    ((C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0)) 
  ↔ a ≥ 1 := by
  sorry

end parabola_intersection_range_l2482_248278


namespace mrs_brown_utility_bill_l2482_248205

def utility_bill_total (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  fifty_bills * 50 + ten_bills * 10

theorem mrs_brown_utility_bill : utility_bill_total 3 2 = 170 := by
  sorry

end mrs_brown_utility_bill_l2482_248205


namespace parabola_intersection_angles_l2482_248236

/-- Parabola C: y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point on the directrix -/
def directrix_point : ℝ × ℝ := (-1, 0)

/-- Line passing through P(m,0) -/
def line (m k : ℝ) (x y : ℝ) : Prop := x = k*y + m

/-- Intersection points of line and parabola -/
def intersection_points (m k : ℝ) : Prop :=
  ∃ (A B : PointOnParabola), A ≠ B ∧ line m k A.x A.y ∧ line m k B.x B.y

/-- Angle between two vectors -/
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_angles (m : ℝ) 
  (h_intersect : ∀ k, intersection_points m k) : 
  (m = 3 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x - directrix_point.1, A.y - directrix_point.2) 
          (B.x - directrix_point.1, B.y - directrix_point.2) < π/2) ∧
  (m = 3 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x - focus.1, A.y - focus.2) 
          (B.x - focus.1, B.y - focus.2) > π/2) ∧
  (m = 4 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x, A.y) (B.x, B.y) = π/2) :=
sorry

end parabola_intersection_angles_l2482_248236


namespace toms_age_ratio_l2482_248267

theorem toms_age_ratio (T N : ℝ) (h1 : T > 0) (h2 : N > 0) 
  (h3 : T - N = 3 * (T - 3 * N)) : T / N = 4 := by
  sorry

end toms_age_ratio_l2482_248267


namespace prime_sum_theorem_l2482_248222

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_sum_theorem (p q : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (h1 : is_prime (7*p + q)) 
  (h2 : is_prime (p*q + 11)) : 
  p^q + q^p = 17 := by sorry

end prime_sum_theorem_l2482_248222


namespace least_four_digit_multiple_of_3_5_7_l2482_248238

theorem least_four_digit_multiple_of_3_5_7 :
  (∀ n : ℕ, n ≥ 1000 ∧ n < 1050 → ¬(3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n)) ∧
  (1050 ≥ 1000 ∧ 3 ∣ 1050 ∧ 5 ∣ 1050 ∧ 7 ∣ 1050) :=
by sorry

end least_four_digit_multiple_of_3_5_7_l2482_248238


namespace expression_simplification_l2482_248211

theorem expression_simplification (q : ℝ) : 
  ((7 * q - 4) - 3 * q * 2) * 4 + (5 - 2 / 2) * (8 * q - 12) = 36 * q - 64 := by
  sorry

end expression_simplification_l2482_248211


namespace distribute_5_3_l2482_248258

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: Distributing 5 distinct objects into 3 distinct groups,
    where each group must contain at least one object, can be done in 150 ways -/
theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end distribute_5_3_l2482_248258


namespace angle_measure_when_complement_and_supplement_are_complementary_l2482_248248

theorem angle_measure_when_complement_and_supplement_are_complementary :
  ∀ x : ℝ,
  (90 - x) + (180 - x) = 90 →
  x = 45 := by
sorry

end angle_measure_when_complement_and_supplement_are_complementary_l2482_248248


namespace amanda_ticket_sales_l2482_248250

/-- The number of tickets Amanda needs to sell in total -/
def total_tickets : ℕ := 150

/-- The number of friends Amanda sells tickets to on the first day -/
def friends : ℕ := 8

/-- The number of tickets each friend buys on the first day -/
def tickets_per_friend : ℕ := 4

/-- The number of tickets Amanda sells on the second day -/
def second_day_tickets : ℕ := 45

/-- The number of tickets Amanda sells on the third day -/
def third_day_tickets : ℕ := 25

/-- The number of tickets Amanda needs to sell on the fourth and fifth day combined -/
def remaining_tickets : ℕ := total_tickets - (friends * tickets_per_friend + second_day_tickets + third_day_tickets)

theorem amanda_ticket_sales : remaining_tickets = 48 := by
  sorry

end amanda_ticket_sales_l2482_248250


namespace arithmetic_sequence_length_l2482_248252

/-- Proves that an arithmetic sequence with first term 2, last term 3007,
    and common difference 5 has 602 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a l d n : ℕ), 
    a = 2 → 
    l = 3007 → 
    d = 5 → 
    l = a + (n - 1) * d → 
    n = 602 := by
  sorry

end arithmetic_sequence_length_l2482_248252


namespace workshop_payment_digit_l2482_248210

-- Define the total payment as 2B0 where B is a single digit
def total_payment (B : Nat) : Nat := 200 + 10 * B

-- Define the condition that B is a single digit
def is_single_digit (B : Nat) : Prop := B ≥ 0 ∧ B ≤ 9

-- Define the condition that the payment is equally divisible among 15 people
def is_equally_divisible (payment : Nat) : Prop := 
  ∃ (individual_payment : Nat), payment = 15 * individual_payment

-- Theorem statement
theorem workshop_payment_digit :
  ∀ B : Nat, is_single_digit B → 
  (is_equally_divisible (total_payment B) ↔ (B = 1 ∨ B = 4)) :=
sorry

end workshop_payment_digit_l2482_248210


namespace smallest_number_divisible_by_five_primes_l2482_248256

theorem smallest_number_divisible_by_five_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) ∧
  n = 2310 :=
by sorry

end smallest_number_divisible_by_five_primes_l2482_248256


namespace min_value_theorem_equality_condition_l2482_248266

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x + 3 / (x^3) ≥ 12 :=
by sorry

theorem equality_condition : 9 * 1 + 3 / (1^3) = 12 :=
by sorry

end min_value_theorem_equality_condition_l2482_248266


namespace vertical_angles_are_congruent_l2482_248281

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define angle congruence
def are_congruent (α β : Angle) : Prop := α = β

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) :
  are_vertical_angles α β → are_congruent α β := by
  sorry

end vertical_angles_are_congruent_l2482_248281
