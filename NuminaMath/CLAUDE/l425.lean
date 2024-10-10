import Mathlib

namespace range_of_a_l425_42568

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) →
  (∃ x : ℝ, x^2 + 4*x + a = 0) →
  a ∈ Set.Icc (Real.exp 1) 4 :=
by sorry

end range_of_a_l425_42568


namespace constant_seq_arithmetic_and_geometric_l425_42521

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A constant sequence with value a -/
def constantSeq (a : ℝ) : Sequence := λ _ => a

/-- An arithmetic sequence -/
def isArithmetic (s : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- A geometric sequence (allowing zero terms) -/
def isGeometric (s : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem constant_seq_arithmetic_and_geometric (a : ℝ) :
  isArithmetic (constantSeq a) ∧ isGeometric (constantSeq a) := by
  sorry

#check constant_seq_arithmetic_and_geometric

end constant_seq_arithmetic_and_geometric_l425_42521


namespace units_digit_of_six_to_seven_l425_42523

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of 6^7 is 6 -/
theorem units_digit_of_six_to_seven :
  unitsDigit (6^7) = 6 := by sorry

end units_digit_of_six_to_seven_l425_42523


namespace num_divisors_not_div_by_5_eq_4_l425_42527

/-- The number of positive divisors of 150 that are not divisible by 5 -/
def num_divisors_not_div_by_5 : ℕ :=
  (Finset.filter (fun d => d ∣ 150 ∧ ¬(5 ∣ d)) (Finset.range 151)).card

/-- 150 has the prime factorization 2 * 3 * 5^2 -/
axiom prime_factorization : 150 = 2 * 3 * 5^2

theorem num_divisors_not_div_by_5_eq_4 : num_divisors_not_div_by_5 = 4 := by
  sorry

end num_divisors_not_div_by_5_eq_4_l425_42527


namespace christophers_to_gabrielas_age_ratio_l425_42594

/-- Proves that the ratio of Christopher's age to Gabriela's age is 2:1 given the conditions -/
theorem christophers_to_gabrielas_age_ratio :
  ∀ (c g : ℕ),
  c = 24 →  -- Christopher is now 24 years old
  c - 9 = 5 * (g - 9) →  -- Nine years ago, Christopher was 5 times as old as Gabriela
  c / g = 2 :=  -- The ratio of Christopher's age to Gabriela's age is 2:1
by
  sorry

#check christophers_to_gabrielas_age_ratio

end christophers_to_gabrielas_age_ratio_l425_42594


namespace equation_equivalence_l425_42547

theorem equation_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 * x = 7 * y) : 
  x / 7 = y / 3 := by
sorry

end equation_equivalence_l425_42547


namespace geometric_sequence_sum_l425_42520

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The main theorem -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l425_42520


namespace round_trip_average_speed_river_boat_average_speed_l425_42576

/-- The average speed of a round trip given upstream and downstream speeds -/
theorem round_trip_average_speed (upstream_speed downstream_speed : ℝ) 
  (upstream_speed_pos : 0 < upstream_speed)
  (downstream_speed_pos : 0 < downstream_speed) :
  (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed) =
  (2 * 6 * 8) / (6 + 8) :=
by sorry

/-- The specific case for the river boat problem -/
theorem river_boat_average_speed :
  (2 * 6 * 8) / (6 + 8) = 48 / 7 :=
by sorry

end round_trip_average_speed_river_boat_average_speed_l425_42576


namespace photo_border_area_l425_42508

/-- The area of the border around a rectangular photograph -/
theorem photo_border_area (photo_height photo_width border_width : ℝ) 
  (h_height : photo_height = 9)
  (h_width : photo_width = 12)
  (h_border : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 162 := by
  sorry

#check photo_border_area

end photo_border_area_l425_42508


namespace distance_between_points_l425_42542

/-- The distance between points (1, 2) and (5, 6) is 4√2 units. -/
theorem distance_between_points : Real.sqrt ((5 - 1)^2 + (6 - 2)^2) = 4 * Real.sqrt 2 := by
  sorry

end distance_between_points_l425_42542


namespace min_abs_a_plus_b_l425_42583

theorem min_abs_a_plus_b (a b : ℤ) (h1 : |a| < |b|) (h2 : |b| ≤ 4) :
  ∃ (m : ℤ), (∀ (x y : ℤ), |x| < |y| → |y| ≤ 4 → m ≤ |x| + y) ∧ m = -4 :=
sorry

end min_abs_a_plus_b_l425_42583


namespace negation_of_universal_statement_l425_42502

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x + 5 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 5 > 0) := by
  sorry

end negation_of_universal_statement_l425_42502


namespace specific_trapezoid_area_l425_42596

/-- An isosceles trapezoid with the given measurements --/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longerBase : ℝ

/-- The area of an isosceles trapezoid --/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- The theorem stating the area of the specific trapezoid --/
theorem specific_trapezoid_area : 
  let t : IsoscelesTrapezoid := { 
    leg := 20,
    diagonal := 25,
    longerBase := 30
  }
  abs (trapezoidArea t - 315.82) < 0.01 := by
  sorry

end specific_trapezoid_area_l425_42596


namespace cardinality_of_star_product_l425_42557

def P : Finset ℕ := {3, 4, 5}
def Q : Finset ℕ := {4, 5, 6, 7}

def star_product (P Q : Finset ℕ) : Finset (ℕ × ℕ) :=
  Finset.product P Q

theorem cardinality_of_star_product :
  Finset.card (star_product P Q) = 12 := by
  sorry

end cardinality_of_star_product_l425_42557


namespace max_product_equals_sum_l425_42577

theorem max_product_equals_sum (H M T : ℤ) : 
  H * M * M * T = H + M + M + T → H * M * M * T ≤ 8 := by
  sorry

end max_product_equals_sum_l425_42577


namespace periodic_coloring_divides_l425_42516

/-- A coloring of the integers -/
def Coloring := ℤ → Bool

/-- A coloring is t-periodic if it repeats every t steps -/
def isPeriodic (c : Coloring) (t : ℕ) : Prop :=
  ∀ x : ℤ, c x = c (x + t)

/-- For a given x, exactly one of x + a₁, ..., x + aₙ is colored -/
def hasUniqueColoredSum (c : Coloring) (a : Fin n → ℕ) : Prop :=
  ∀ x : ℤ, ∃! i : Fin n, c (x + a i)

theorem periodic_coloring_divides (n : ℕ) (t : ℕ) (a : Fin n → ℕ) (h_a : StrictMono a) 
    (c : Coloring) (h_periodic : isPeriodic c t) (h_unique : hasUniqueColoredSum c a) : 
    n ∣ t := by sorry

end periodic_coloring_divides_l425_42516


namespace right_triangle_acute_angles_l425_42571

theorem right_triangle_acute_angles (α β : Real) : 
  -- Conditions
  α + β = 90 →  -- Sum of acute angles in a right triangle is 90°
  α = 40 →      -- One acute angle is 40°
  -- Conclusion
  β = 50 :=     -- The other acute angle is 50°
by sorry

end right_triangle_acute_angles_l425_42571


namespace jesse_has_21_bananas_l425_42566

/-- The number of bananas Jesse has -/
def jesse_bananas : ℕ := 21

/-- The number of friends Jesse shares the bananas with -/
def num_friends : ℕ := 3

/-- The number of bananas each friend gets when Jesse shares his bananas -/
def bananas_per_friend : ℕ := 7

/-- Theorem stating that Jesse has 21 bananas -/
theorem jesse_has_21_bananas : 
  jesse_bananas = num_friends * bananas_per_friend := by
  sorry

end jesse_has_21_bananas_l425_42566


namespace new_light_wattage_l425_42543

theorem new_light_wattage (old_wattage : ℝ) (increase_percentage : ℝ) (new_wattage : ℝ) :
  old_wattage = 80 →
  increase_percentage = 0.25 →
  new_wattage = old_wattage * (1 + increase_percentage) →
  new_wattage = 100 :=
by
  sorry

end new_light_wattage_l425_42543


namespace fraction_value_l425_42597

theorem fraction_value (a b : ℝ) (h : a + 1/b = 2/a + 2*b ∧ a + 1/b ≠ 0) : a/b = 2 := by
  sorry

end fraction_value_l425_42597


namespace lawn_mowing_difference_l425_42553

/-- The difference between spring and summer lawn mowing counts -/
theorem lawn_mowing_difference (spring_count summer_count : ℕ) 
  (h1 : spring_count = 8) 
  (h2 : summer_count = 5) : 
  spring_count - summer_count = 3 := by
  sorry

end lawn_mowing_difference_l425_42553


namespace light_intensity_reduction_l425_42572

/-- Given light with original intensity a passing through n pieces of glass,
    each reducing intensity by 10%, calculate the final intensity -/
def final_intensity (a : ℝ) (n : ℕ) : ℝ :=
  a * (0.9 ^ n)

/-- Theorem: Light with original intensity a passing through 3 pieces of glass,
    each reducing intensity by 10%, results in a final intensity of 0.729a -/
theorem light_intensity_reduction (a : ℝ) :
  final_intensity a 3 = 0.729 * a := by
  sorry

end light_intensity_reduction_l425_42572


namespace root_implies_b_value_l425_42565

theorem root_implies_b_value (a b : ℚ) :
  (2 + Real.sqrt 5 : ℝ) ^ 3 + a * (2 + Real.sqrt 5 : ℝ) ^ 2 + b * (2 + Real.sqrt 5 : ℝ) - 20 = 0 →
  b = -24 := by
sorry

end root_implies_b_value_l425_42565


namespace smallest_k_for_sum_and_product_existence_of_solution_smallest_k_is_four_l425_42584

theorem smallest_k_for_sum_and_product (k : ℝ) : 
  (k > 0 ∧ 
   ∃ a b : ℝ, a + b = k ∧ a * b = k) → 
  k ≥ 4 :=
by sorry

theorem existence_of_solution : 
  ∃ k a b : ℝ, k > 0 ∧ a + b = k ∧ a * b = k ∧ k = 4 :=
by sorry

theorem smallest_k_is_four : 
  ∃! k : ℝ, k > 0 ∧ 
  (∃ a b : ℝ, a + b = k ∧ a * b = k) ∧
  (∀ k' : ℝ, k' > 0 → (∃ a b : ℝ, a + b = k' ∧ a * b = k') → k' ≥ k) ∧
  k = 4 :=
by sorry

end smallest_k_for_sum_and_product_existence_of_solution_smallest_k_is_four_l425_42584


namespace square_side_length_l425_42509

/-- Given a rectangle with width 36 cm and length 64 cm, and a square whose perimeter
    equals the rectangle's perimeter, prove that the side length of the square is 50 cm. -/
theorem square_side_length (rectangle_width rectangle_length : ℝ)
                            (square_side : ℝ)
                            (h1 : rectangle_width = 36)
                            (h2 : rectangle_length = 64)
                            (h3 : 4 * square_side = 2 * (rectangle_width + rectangle_length)) :
  square_side = 50 := by
  sorry

end square_side_length_l425_42509


namespace quadratic_equation_result_l425_42589

theorem quadratic_equation_result (x : ℝ) : 
  2 * x^2 - 5 = 11 → 
  (4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2) ∨ 
  (4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2) := by
sorry

end quadratic_equation_result_l425_42589


namespace largest_term_is_115_div_3_l425_42560

/-- An arithmetic sequence of 5 terms satisfying specific conditions -/
structure ArithmeticSequence where
  terms : Fin 5 → ℚ
  is_arithmetic : ∀ i j k : Fin 5, terms k - terms j = terms j - terms i
  sum_is_100 : (Finset.univ.sum terms) = 100
  ratio_condition : (terms 2 + terms 3 + terms 4) = (1/7) * (terms 0 + terms 1)

/-- The largest term in the arithmetic sequence is 115/3 -/
theorem largest_term_is_115_div_3 (seq : ArithmeticSequence) : seq.terms 4 = 115/3 := by
  sorry

end largest_term_is_115_div_3_l425_42560


namespace j_percentage_less_than_p_l425_42586

/-- Given t = 6.25, t is t% less than p, and j is 20% less than t, prove j is 25% less than p -/
theorem j_percentage_less_than_p (t p j : ℝ) : 
  t = 6.25 →
  t = p * (100 - t) / 100 →
  j = t * 0.8 →
  j = p * 0.75 := by
  sorry

end j_percentage_less_than_p_l425_42586


namespace sum_of_roots_l425_42548

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → a * (a - 6) = 7 → b * (b - 6) = 7 → a + b = 6 := by
  sorry

end sum_of_roots_l425_42548


namespace bobby_total_pieces_l425_42512

/-- The total number of candy and chocolate pieces Bobby ate -/
def total_pieces (initial_candy : ℕ) (additional_candy : ℕ) (chocolate : ℕ) : ℕ :=
  initial_candy + additional_candy + chocolate

/-- Theorem stating that Bobby ate 51 pieces of candy and chocolate in total -/
theorem bobby_total_pieces :
  total_pieces 33 4 14 = 51 := by
  sorry

end bobby_total_pieces_l425_42512


namespace min_value_on_circle_l425_42585

theorem min_value_on_circle (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 16 = 0 → ∃ (m : ℝ), m = 4 ∧ ∀ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 16 = 0 → x^2 + y^2 ≥ m :=
by sorry

end min_value_on_circle_l425_42585


namespace mara_crayon_count_l425_42592

theorem mara_crayon_count : ∀ (mara_crayons : ℕ),
  (mara_crayons : ℚ) * (1 / 10 : ℚ) + (50 : ℚ) * (1 / 5 : ℚ) = 14 →
  mara_crayons = 40 := by
  sorry

end mara_crayon_count_l425_42592


namespace acute_triangle_condition_l425_42510

/-- 
Given a unit circle with diameter AB, where A(-1, 0) and B(1, 0),
and a point D(x, 0) on AB, prove that AD, BD, and CD form an acute triangle
if and only if x is in the open interval (2 - √5, √5 - 2),
where C is the point where DC ⊥ AB intersects the circle.
-/
theorem acute_triangle_condition (x : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (x, 0)
  let C : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
  let AD := Real.sqrt ((x + 1)^2)
  let BD := Real.sqrt ((1 - x)^2)
  let CD := Real.sqrt (1 - x^2)
  (AD^2 + BD^2 > CD^2 ∧ AD^2 + CD^2 > BD^2 ∧ BD^2 + CD^2 > AD^2) ↔ 
  (x > 2 - Real.sqrt 5 ∧ x < Real.sqrt 5 - 2) :=
by sorry


end acute_triangle_condition_l425_42510


namespace F_of_4_f_of_5_equals_174_l425_42529

-- Define the function f
def f (a : ℝ) : ℝ := 3 * a - 6

-- Define the function F
def F (a b : ℝ) : ℝ := 2 * b^2 + 3 * a

-- Theorem statement
theorem F_of_4_f_of_5_equals_174 : F 4 (f 5) = 174 := by
  sorry

end F_of_4_f_of_5_equals_174_l425_42529


namespace tiling_impossibility_l425_42507

/-- Represents a rectangular area that can be tiled. -/
structure TileableArea where
  width : ℕ
  height : ℕ

/-- Represents the count of each type of tile. -/
structure TileCount where
  two_by_two : ℕ
  one_by_four : ℕ

/-- Checks if a given area can be tiled with the given tile counts. -/
def can_tile (area : TileableArea) (tiles : TileCount) : Prop :=
  2 * tiles.two_by_two + 4 * tiles.one_by_four = area.width * area.height

/-- Theorem stating that if an area can be tiled, it becomes impossible
    to tile after replacing one 2x2 tile with a 1x4 tile. -/
theorem tiling_impossibility (area : TileableArea) (initial_tiles : TileCount) :
  can_tile area initial_tiles →
  ¬can_tile area { two_by_two := initial_tiles.two_by_two - 1,
                   one_by_four := initial_tiles.one_by_four + 1 } :=
by sorry

end tiling_impossibility_l425_42507


namespace garage_sale_necklace_cost_l425_42525

/-- The cost of each necklace in Isabel's garage sale --/
def cost_per_necklace (total_necklaces : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / total_necklaces

/-- Theorem stating that the cost per necklace is $6 --/
theorem garage_sale_necklace_cost :
  cost_per_necklace 6 36 = 6 := by
  sorry

end garage_sale_necklace_cost_l425_42525


namespace solve_percentage_problem_l425_42531

theorem solve_percentage_problem (x : ℝ) : (0.7 * x = (1/3) * x + 110) → x = 300 := by
  sorry

end solve_percentage_problem_l425_42531


namespace quadratic_intersection_theorem_l425_42579

def line_l (x y : ℝ) : Prop := y = 4

def quadratic_function (x a : ℝ) : ℝ :=
  (x - a)^2 + (x - 2*a)^2 + (x - 3*a)^2 - 2*a^2 + a

def has_two_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    line_l x₁ (quadratic_function x₁ a) ∧
    line_l x₂ (quadratic_function x₂ a)

def axis_of_symmetry (a : ℝ) : ℝ := 2 * a

theorem quadratic_intersection_theorem (a : ℝ) :
  has_two_intersections a ∧ axis_of_symmetry a > 0 → 0 < a ∧ a < 4 :=
sorry

end quadratic_intersection_theorem_l425_42579


namespace sum_floor_equality_l425_42539

theorem sum_floor_equality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + b^2 = 2008) (h2 : c^2 + d^2 = 2008) (h3 : a * c = 1000) (h4 : b * d = 1000) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end sum_floor_equality_l425_42539


namespace limo_cost_per_hour_l425_42559

/-- Calculates the cost of a limo per hour given prom expenses -/
theorem limo_cost_per_hour 
  (ticket_cost : ℝ) 
  (dinner_cost : ℝ) 
  (tip_percentage : ℝ) 
  (limo_hours : ℝ) 
  (total_cost : ℝ) 
  (h1 : ticket_cost = 100)
  (h2 : dinner_cost = 120)
  (h3 : tip_percentage = 0.3)
  (h4 : limo_hours = 6)
  (h5 : total_cost = 836) :
  (total_cost - (2 * ticket_cost + dinner_cost + tip_percentage * dinner_cost)) / limo_hours = 80 :=
by sorry

end limo_cost_per_hour_l425_42559


namespace least_addition_for_divisibility_l425_42551

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), (1056 + x) % 23 = 0 ∧ ∀ (y : ℕ), y < x → (1056 + y) % 23 ≠ 0 :=
by
  -- The proof would go here
  sorry

end least_addition_for_divisibility_l425_42551


namespace cube_root_fraction_equality_l425_42518

theorem cube_root_fraction_equality : 
  (((5 : ℝ) / 6 * 20.25) ^ (1/3 : ℝ)) = (3 * (5 ^ (2/3 : ℝ))) / 2 := by
  sorry

end cube_root_fraction_equality_l425_42518


namespace T_is_Y_shape_l425_42588

/-- The set T of points (x, y) in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 6 < 5) ∨
               (5 = y - 6 ∧ x + 3 < 5) ∨
               (x + 3 = y - 6 ∧ 5 < x + 3)}

/-- The common start point of the "Y" shape -/
def commonPoint : ℝ × ℝ := (2, 11)

/-- The vertical line segment of the "Y" shape -/
def verticalSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 2 ∧ p.2 < 11}

/-- The horizontal line segment of the "Y" shape -/
def horizontalSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 11 ∧ p.1 < 2}

/-- The diagonal ray of the "Y" shape -/
def diagonalRay : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 > 2}

theorem T_is_Y_shape :
  T = verticalSegment ∪ horizontalSegment ∪ diagonalRay ∧
  commonPoint ∈ T ∧
  commonPoint ∈ verticalSegment ∧
  commonPoint ∈ horizontalSegment ∧
  commonPoint ∈ diagonalRay :=
sorry

end T_is_Y_shape_l425_42588


namespace smallest_x_squared_is_2135_l425_42578

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  x : ℝ
  has_circle : Bool
  has_tangent_line : Bool

/-- The smallest possible value of x^2 for the given trapezoid -/
def smallest_x_squared (t : IsoscelesTrapezoid) : ℝ := 2135

/-- Theorem stating the smallest possible value of x^2 for the specific trapezoid -/
theorem smallest_x_squared_is_2135 (t : IsoscelesTrapezoid) 
  (h1 : t.AB = 122) 
  (h2 : t.CD = 26) 
  (h3 : t.has_circle = true) 
  (h4 : t.has_tangent_line = true) : 
  smallest_x_squared t = 2135 := by
  sorry

end smallest_x_squared_is_2135_l425_42578


namespace james_initial_milk_l425_42513

def ounces_drank : ℕ := 13
def ounces_per_gallon : ℕ := 128
def ounces_left : ℕ := 371

def initial_gallons : ℚ :=
  (ounces_left + ounces_drank) / ounces_per_gallon

theorem james_initial_milk : initial_gallons = 3 := by
  sorry

end james_initial_milk_l425_42513


namespace meaningful_sqrt_range_l425_42562

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 / (x - 1))) → x > 1 := by
sorry

end meaningful_sqrt_range_l425_42562


namespace b_received_15_pencils_l425_42599

/-- The number of pencils each student received -/
structure PencilDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The conditions of the pencil distribution problem -/
def ValidDistribution (p : PencilDistribution) : Prop :=
  p.a + p.b + p.c + p.d = 53 ∧
  (max p.a (max p.b (max p.c p.d))) - (min p.a (min p.b (min p.c p.d))) ≤ 5 ∧
  p.a + p.b = 2 * p.c ∧
  p.c + p.b = 2 * p.d

/-- The theorem stating that B received 15 pencils -/
theorem b_received_15_pencils (p : PencilDistribution) (h : ValidDistribution p) : p.b = 15 := by
  sorry

end b_received_15_pencils_l425_42599


namespace xy_max_value_l425_42556

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 2) :
  xy ≤ (1 : ℝ) / 2 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 2 ∧ x₀ * y₀ = (1 : ℝ) / 2 :=
sorry

end xy_max_value_l425_42556


namespace ellipse_properties_l425_42500

/-- Definition of an ellipse passing through a point with given foci -/
def is_ellipse_through_point (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f2.1 - f1.1)^2 + (f2.2 - f1.2)^2) / 2
  ∃ a : ℝ, a > c ∧ d1 + d2 = 2 * a

/-- The equation of an ellipse in standard form -/
def ellipse_equation (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties :
  let f1 : ℝ × ℝ := (0, 0)
  let f2 : ℝ × ℝ := (0, 8)
  let p : ℝ × ℝ := (7, 4)
  let a : ℝ := 8 * Real.sqrt 2
  let b : ℝ := 8 * Real.sqrt 7
  let h : ℝ := 0
  let k : ℝ := 4
  is_ellipse_through_point f1 f2 p →
  (∀ x y : ℝ, ellipse_equation a b h k x y ↔ 
    ((x - 0)^2 / (8 * Real.sqrt 2)^2 + (y - 4)^2 / (8 * Real.sqrt 7)^2 = 1)) :=
by sorry

end ellipse_properties_l425_42500


namespace square_area_reduction_l425_42526

theorem square_area_reduction (S1_area : ℝ) (S1_area_eq : S1_area = 25) : 
  let S1_side := Real.sqrt S1_area
  let S2_side := S1_side / Real.sqrt 2
  let S3_side := S2_side / Real.sqrt 2
  S3_side ^ 2 = 6.25 := by
  sorry

end square_area_reduction_l425_42526


namespace right_triangle_cone_rotation_l425_42552

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    with volume 800π cm³ and rotating about leg b produces a cone with volume 1920π cm³,
    then the hypotenuse length is 26 cm. -/
theorem right_triangle_cone_rotation (a b : ℝ) :
  a > 0 ∧ b > 0 →
  (1 / 3 : ℝ) * Real.pi * a * b^2 = 800 * Real.pi →
  (1 / 3 : ℝ) * Real.pi * b * a^2 = 1920 * Real.pi →
  Real.sqrt (a^2 + b^2) = 26 := by
  sorry


end right_triangle_cone_rotation_l425_42552


namespace river_improvement_equation_l425_42519

theorem river_improvement_equation (x : ℝ) (h : x > 0) : 
  (4800 / x) - (4800 / (x + 200)) = 4 ↔ 
  (∃ (planned_days actual_days : ℝ),
    planned_days = 4800 / x ∧
    actual_days = 4800 / (x + 200) ∧
    planned_days - actual_days = 4) :=
by sorry

end river_improvement_equation_l425_42519


namespace certain_number_problem_l425_42573

theorem certain_number_problem : ∃ x : ℝ, 0.12 * x - 0.1 * 14.2 = 1.484 ∧ x = 24.2 := by
  sorry

end certain_number_problem_l425_42573


namespace last_card_in_box_three_l425_42564

/-- Represents the number of boxes --/
def num_boxes : ℕ := 7

/-- Represents the total number of cards --/
def total_cards : ℕ := 2015

/-- Represents the length of one complete cycle --/
def cycle_length : ℕ := 12

/-- Calculates the box number for a given card number --/
def box_for_card (card_num : ℕ) : ℕ :=
  let position_in_cycle := card_num % cycle_length
  if position_in_cycle ≤ num_boxes then
    position_in_cycle
  else
    num_boxes - (position_in_cycle - num_boxes)

/-- Theorem stating that the last card (2015th) will be placed in box 3 --/
theorem last_card_in_box_three :
  box_for_card total_cards = 3 := by
  sorry

end last_card_in_box_three_l425_42564


namespace ellipse_transformation_l425_42503

/-- Given an ellipse with equation x²/6 + y² = 1, prove that compressing
    the x-coordinates to 1/2 of their original value and stretching the
    y-coordinates to twice their original value results in a curve with
    equation 2x²/3 + y²/4 = 1. -/
theorem ellipse_transformation (x y : ℝ) :
  (x^2 / 6 + y^2 = 1) →
  (∃ x' y' : ℝ, x' = x / 2 ∧ y' = 2 * y ∧ 2 * x'^2 / 3 + y'^2 / 4 = 1) :=
sorry

end ellipse_transformation_l425_42503


namespace gold_silver_board_theorem_l425_42593

/-- A board configuration with gold and silver cells -/
structure Board :=
  (size : Nat)
  (is_gold : Fin size → Fin size → Bool)

/-- Count gold cells in a rectangle -/
def count_gold (b : Board) (x y w h : Nat) : Nat :=
  (Finset.range w).sum (λ i =>
    (Finset.range h).sum (λ j =>
      if b.is_gold ⟨x + i, sorry⟩ ⟨y + j, sorry⟩ then 1 else 0))

/-- Property that each 3x3 square has A gold cells -/
def three_by_three_property (b : Board) (A : Nat) : Prop :=
  ∀ x y, x + 3 ≤ b.size → y + 3 ≤ b.size →
    count_gold b x y 3 3 = A

/-- Property that each 2x4 or 4x2 rectangle has Z gold cells -/
def two_by_four_property (b : Board) (Z : Nat) : Prop :=
  (∀ x y, x + 2 ≤ b.size → y + 4 ≤ b.size →
    count_gold b x y 2 4 = Z) ∧
  (∀ x y, x + 4 ≤ b.size → y + 2 ≤ b.size →
    count_gold b x y 4 2 = Z)

/-- The main theorem -/
theorem gold_silver_board_theorem :
  ∀ (b : Board) (A Z : Nat),
    b.size = 2016 →
    three_by_three_property b A →
    two_by_four_property b Z →
    ((A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8)) :=
sorry

end gold_silver_board_theorem_l425_42593


namespace simplify_fraction_l425_42558

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (75 * b^3) = 2/5 := by
  sorry

end simplify_fraction_l425_42558


namespace fraction_sum_equals_one_l425_42535

theorem fraction_sum_equals_one (a : ℝ) (h : a ≠ -1) :
  (1 : ℝ) / (a + 1) + a / (a + 1) = 1 := by
  sorry

end fraction_sum_equals_one_l425_42535


namespace cuboid_volume_doubled_l425_42569

/-- Theorem: Doubling dimensions of a cuboid results in 8 times the original volume -/
theorem cuboid_volume_doubled (l w h : ℝ) (l_pos : 0 < l) (w_pos : 0 < w) (h_pos : 0 < h) :
  (2 * l) * (2 * w) * (2 * h) = 8 * (l * w * h) := by
  sorry

#check cuboid_volume_doubled

end cuboid_volume_doubled_l425_42569


namespace f_monotonic_range_l425_42534

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the property of being monotonic on an interval
def IsMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f x > f y)

-- Theorem statement
theorem f_monotonic_range (m : ℝ) :
  IsMonotonicOn f m (m + 4) → m ∈ Set.Iic (-5) ∪ Set.Ici 2 :=
sorry

end f_monotonic_range_l425_42534


namespace range_of_a_l425_42504

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) → 
  a ∈ Set.Iic (-2) ∪ {1} :=
sorry

end range_of_a_l425_42504


namespace usb_storage_capacity_l425_42561

/-- Represents the capacity of a storage device in gigabytes -/
def StorageCapacityGB : ℕ := 2

/-- Represents the size of one gigabyte in megabytes -/
def GBtoMB : ℕ := 2^10

/-- Represents the file size of each photo in megabytes -/
def PhotoSizeMB : ℕ := 16

/-- Calculates the number of photos that can be stored -/
def NumberOfPhotos : ℕ := 2^7

theorem usb_storage_capacity :
  StorageCapacityGB * GBtoMB / PhotoSizeMB = NumberOfPhotos :=
sorry

end usb_storage_capacity_l425_42561


namespace vector_operation_l425_42545

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![2, -2]

theorem vector_operation : 2 • a - b = ![2, 4] := by sorry

end vector_operation_l425_42545


namespace set_equality_unordered_elements_l425_42532

theorem set_equality_unordered_elements : 
  let M : Set ℕ := {4, 5}
  let N : Set ℕ := {5, 4}
  M = N :=
by sorry

end set_equality_unordered_elements_l425_42532


namespace angle_point_cosine_l425_42517

/-- Given an angle α and a real number a, proves that if the terminal side of α
    passes through point P(3a, 4) and cos α = -3/5, then a = -1. -/
theorem angle_point_cosine (α : Real) (a : Real) : 
  (∃ r : Real, r > 0 ∧ 3 * a = r * Real.cos α ∧ 4 = r * Real.sin α) → 
  Real.cos α = -3/5 → 
  a = -1 := by
  sorry

end angle_point_cosine_l425_42517


namespace fraction_simplification_l425_42533

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 2) :
  (a + 1) / (a^2 - 1) / ((a^2 - 4) / (a^2 + a - 2)) - (1 - a) / (a - 2) = a / (a - 2) := by
  sorry

end fraction_simplification_l425_42533


namespace dalmatian_spots_l425_42554

theorem dalmatian_spots (b p : ℕ) (h1 : b = 2 * p - 1) (h2 : b + p = 59) : b = 39 := by
  sorry

end dalmatian_spots_l425_42554


namespace sufficient_not_necessary_l425_42595

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → x^2 + x - 2 > 0) ∧
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ 1) :=
by sorry

end sufficient_not_necessary_l425_42595


namespace rolling_coin_curve_length_l425_42501

/-- The length of the curve traced by the center of a rolling coin -/
theorem rolling_coin_curve_length 
  (coin_circumference : ℝ) 
  (quadrilateral_perimeter : ℝ) : 
  coin_circumference = 5 →
  quadrilateral_perimeter = 20 →
  (curve_length : ℝ) = quadrilateral_perimeter + coin_circumference →
  curve_length = 25 :=
by sorry

end rolling_coin_curve_length_l425_42501


namespace sum_of_reciprocals_l425_42528

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h_sum : x + y = 3 * x * y) (h_diff : x - y = 1) :
  1 / x + 1 / y = Real.sqrt 13 + 2 := by
  sorry

end sum_of_reciprocals_l425_42528


namespace min_cuts_for_cube_division_l425_42515

/-- Represents a three-dimensional cube --/
structure Cube where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the process of cutting a cube --/
def cut_cube (initial : Cube) (final_size : ℕ) (allow_rearrange : Bool) : ℕ :=
  sorry

/-- Theorem: The minimum number of cuts to divide a 3x3x3 cube into 27 1x1x1 cubes is 6 --/
theorem min_cuts_for_cube_division :
  let initial_cube : Cube := ⟨3, 3, 3⟩
  let final_size : ℕ := 1
  let num_final_cubes : ℕ := 27
  let allow_rearrange : Bool := true
  (cut_cube initial_cube final_size allow_rearrange = 6) ∧
  (∀ n : ℕ, n < 6 → cut_cube initial_cube final_size allow_rearrange ≠ n) :=
by sorry

end min_cuts_for_cube_division_l425_42515


namespace graduating_class_boys_count_l425_42536

theorem graduating_class_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 466 →
  diff = 212 →
  boys + (boys + diff) = total →
  boys = 127 := by
sorry

end graduating_class_boys_count_l425_42536


namespace ant_distance_theorem_l425_42567

def ant_movement : List (ℝ × ℝ) := [(-7, 0), (0, 5), (3, 0), (0, -2), (9, 0), (0, -2), (-1, 0), (0, -1)]

def total_displacement (movements : List (ℝ × ℝ)) : ℝ × ℝ :=
  movements.foldl (λ (acc : ℝ × ℝ) (move : ℝ × ℝ) => (acc.1 + move.1, acc.2 + move.2)) (0, 0)

theorem ant_distance_theorem :
  let final_position := total_displacement ant_movement
  Real.sqrt (final_position.1 ^ 2 + final_position.2 ^ 2) = 4 := by
  sorry

end ant_distance_theorem_l425_42567


namespace doctors_lawyers_ratio_l425_42511

theorem doctors_lawyers_ratio (d l : ℕ) (h_total : d + l > 0) :
  (45 * d + 55 * l) / (d + l) = 47 →
  d = 4 * l :=
by
  sorry

end doctors_lawyers_ratio_l425_42511


namespace power_division_equality_l425_42538

theorem power_division_equality (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end power_division_equality_l425_42538


namespace cosine_identity_l425_42580

theorem cosine_identity (z : ℂ) (α : ℝ) (h : z + 1/z = 2 * Real.cos α) :
  ∀ n : ℕ, z^n + 1/z^n = 2 * Real.cos (n * α) := by
  sorry

end cosine_identity_l425_42580


namespace log_one_half_decreasing_l425_42550

-- Define the logarithm function with base a
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our specific function f(x) = log_(1/2)(x)
noncomputable def f (x : ℝ) : ℝ := log (1/2) x

-- State the theorem
theorem log_one_half_decreasing :
  0 < (1/2 : ℝ) ∧ (1/2 : ℝ) < 1 →
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x < y → f y < f x :=
sorry

end log_one_half_decreasing_l425_42550


namespace inverse_88_mod_89_l425_42505

theorem inverse_88_mod_89 : ∃ x : ℕ, x ≤ 88 ∧ (88 * x) % 89 = 1 :=
by
  -- The proof would go here
  sorry

end inverse_88_mod_89_l425_42505


namespace range_of_a_l425_42506

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 := by
  sorry

end range_of_a_l425_42506


namespace initial_books_l425_42537

theorem initial_books (initial sold bought final : ℕ) : 
  sold = 94 →
  bought = 150 →
  final = 58 →
  initial - sold + bought = final →
  initial = 2 :=
by sorry

end initial_books_l425_42537


namespace lemonade_recipe_correct_l425_42581

/-- Represents the ratio of ingredients in the lemonade mixture -/
structure LemonadeRatio where
  water : ℕ
  lemon_juice : ℕ

/-- Converts gallons to quarts -/
def gallons_to_quarts (gallons : ℕ) : ℕ := 4 * gallons

/-- Calculates the amount of each ingredient needed for a given total volume -/
def ingredient_amount (ratio : LemonadeRatio) (total_volume : ℕ) (ingredient : ℕ) : ℕ :=
  (ingredient * total_volume) / (ratio.water + ratio.lemon_juice)

theorem lemonade_recipe_correct (ratio : LemonadeRatio) (total_gallons : ℕ) :
  ratio.water = 5 →
  ratio.lemon_juice = 3 →
  total_gallons = 2 →
  let total_quarts := gallons_to_quarts total_gallons
  ingredient_amount ratio total_quarts ratio.water = 5 ∧
  ingredient_amount ratio total_quarts ratio.lemon_juice = 3 := by
  sorry

end lemonade_recipe_correct_l425_42581


namespace toothpick_pattern_l425_42582

/-- Given an arithmetic sequence with first term 4 and common difference 4,
    the 150th term is equal to 600. -/
theorem toothpick_pattern (a : ℕ) (d : ℕ) (n : ℕ) :
  a = 4 → d = 4 → n = 150 → a + (n - 1) * d = 600 := by
  sorry

end toothpick_pattern_l425_42582


namespace complex_fraction_equality_l425_42524

theorem complex_fraction_equality : 
  let z₁ : ℂ := 1 - I
  let z₂ : ℂ := 1 + I
  z₁ / (z₂ * I) = -2 * I := by
sorry

end complex_fraction_equality_l425_42524


namespace complex_simplification_l425_42570

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end complex_simplification_l425_42570


namespace min_cut_length_for_non_triangle_l425_42575

def cannot_form_triangle (a b c : ℝ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem min_cut_length_for_non_triangle : ∃ (x : ℝ),
  (x > 0) ∧
  (cannot_form_triangle (9 - x) (12 - x) (15 - x)) ∧
  (∀ y, 0 < y ∧ y < x → ¬(cannot_form_triangle (9 - y) (12 - y) (15 - y))) ∧
  x = 6 := by
sorry

end min_cut_length_for_non_triangle_l425_42575


namespace math_club_minimum_size_l425_42574

theorem math_club_minimum_size :
  ∀ (boys girls : ℕ),
  (boys : ℝ) / (boys + girls : ℝ) > 0.6 →
  girls = 5 →
  boys + girls ≥ 13 ∧
  ∀ (total : ℕ), total < 13 →
    ¬(∃ (b g : ℕ), b + g = total ∧ (b : ℝ) / (total : ℝ) > 0.6 ∧ g = 5) :=
by
  sorry

end math_club_minimum_size_l425_42574


namespace expression_value_l425_42514

theorem expression_value : 
  (7 - (540 : ℚ) / 9) - (5 - (330 : ℚ) * 2 / 11) + (2 - (260 : ℚ) * 3 / 13) = -56 := by
  sorry

end expression_value_l425_42514


namespace gcd_85_100_l425_42530

theorem gcd_85_100 : Nat.gcd 85 100 = 5 := by
  sorry

end gcd_85_100_l425_42530


namespace apples_per_pie_l425_42590

/-- Given a box of apples, calculate the weight of apples needed per pie -/
theorem apples_per_pie (total_weight : ℝ) (num_pies : ℕ) : 
  total_weight = 120 → num_pies = 15 → (total_weight / 2) / num_pies = 4 := by
  sorry

end apples_per_pie_l425_42590


namespace function_increasing_decreasing_implies_m_range_l425_42563

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- State the theorem
theorem function_increasing_decreasing_implies_m_range :
  ∀ m : ℝ, 
  (∀ x ≥ 2, ∀ y ≥ 2, x < y → f m x < f m y) ∧ 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → f m x > f m y) →
  8 ≤ m ∧ m ≤ 16 :=
sorry

end function_increasing_decreasing_implies_m_range_l425_42563


namespace rectangle_square_cut_l425_42587

theorem rectangle_square_cut (m n : ℕ) (hm : m > 2) (hn : n > 2) :
  (m - 2) * (n - 2) = 8 ↔
  (2 * (m + n) - 4 = m * n) ∧ (m * n - 4 = 2 * (m + n)) :=
by sorry

end rectangle_square_cut_l425_42587


namespace digit_configuration_impossible_l425_42544

/-- Represents a configuration of digits on a shape with 6 segments -/
structure DigitConfiguration :=
  (digits : Finset ℕ)
  (segments : Finset (Finset ℕ))

/-- The property that all segments have the same sum -/
def has_equal_segment_sums (config : DigitConfiguration) : Prop :=
  ∃ (sum : ℕ), ∀ segment ∈ config.segments, (segment.sum id = sum)

/-- The main theorem stating the impossibility of the configuration -/
theorem digit_configuration_impossible : 
  ¬ ∃ (config : DigitConfiguration), 
    (config.digits = Finset.range 10) ∧ 
    (config.segments.card = 6) ∧
    (∀ segment ∈ config.segments, segment.card = 3) ∧
    (has_equal_segment_sums config) :=
sorry

end digit_configuration_impossible_l425_42544


namespace minimum_planting_cost_l425_42541

/-- Represents the dimensions of a rectangular region -/
structure Region where
  width : ℝ
  height : ℝ

/-- Represents a type of flower with its cost -/
structure Flower where
  name : String
  cost : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.width * r.height

/-- Calculates the cost of planting a flower in a region -/
def plantingCost (f : Flower) (r : Region) : ℝ := f.cost * area r

/-- The flower bed configuration -/
def flowerBed : Region := { width := 11, height := 6 }

/-- The vertical strip -/
def verticalStrip : Region := { width := 3, height := 6 }

/-- The horizontal strip -/
def horizontalStrip : Region := { width := 11, height := 2 }

/-- The overlap region between vertical and horizontal strips -/
def overlapRegion : Region := { width := 3, height := 2 }

/-- The remaining region -/
def remainingRegion : Region :=
  { width := flowerBed.width - verticalStrip.width,
    height := flowerBed.height - horizontalStrip.height }

/-- The available flower types -/
def flowers : List Flower :=
  [{ name := "Easter Lily", cost := 3 },
   { name := "Dahlia", cost := 2.5 },
   { name := "Canna", cost := 2 }]

/-- Theorem: The minimum cost for planting the flowers is $157 -/
theorem minimum_planting_cost :
  plantingCost (flowers[2]) remainingRegion +
  plantingCost (flowers[1]) verticalStrip +
  plantingCost (flowers[0]) { width := horizontalStrip.width - verticalStrip.width,
                              height := horizontalStrip.height } = 157 := by
  sorry


end minimum_planting_cost_l425_42541


namespace f_extremum_f_range_of_a_l425_42549

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * (x - 1) + b * Real.exp x) / Real.exp x

-- Part 1
theorem f_extremum :
  let a : ℝ := -1
  let b : ℝ := 0
  (∃ x : ℝ, ∀ y : ℝ, f a b y ≥ f a b x) ∧
  (∀ x : ℝ, f a b x ≥ -1 / Real.exp 2) ∧
  (¬ ∃ M : ℝ, ∀ x : ℝ, f a b x ≤ M) := by sorry

-- Part 2
theorem f_range_of_a :
  let b : ℝ := 1
  (∀ a : ℝ, (∀ x : ℝ, f a b x ≠ 0) → a ∈ Set.Ioo (-Real.exp 2) 0) ∧
  (∀ a : ℝ, a ∈ Set.Ioo (-Real.exp 2) 0 → (∀ x : ℝ, f a b x ≠ 0)) := by sorry

end

end f_extremum_f_range_of_a_l425_42549


namespace goldbach_conjecture_negation_l425_42546

-- Define the Goldbach Conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- State the theorem
theorem goldbach_conjecture_negation :
  ¬goldbach_conjecture ↔ ∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q :=
by
  sorry

end goldbach_conjecture_negation_l425_42546


namespace domino_swap_incorrect_l425_42522

/-- Represents a domino with a value from 0 to 9 -/
def Domino : Type := Fin 10

/-- Represents a multiplication problem with 5 dominoes -/
structure DominoMultiplication :=
  (d1 d2 d3 d4 d5 : Domino)

/-- Checks if the domino multiplication is correct -/
def isCorrectMultiplication (dm : DominoMultiplication) : Prop :=
  (dm.d1.val * 10 + dm.d2.val) * dm.d3.val = dm.d4.val * 10 + dm.d5.val

/-- Swaps two dominoes in the multiplication -/
def swapDominoes (dm : DominoMultiplication) (i j : Fin 5) : DominoMultiplication :=
  match i, j with
  | 0, 1 => { d1 := dm.d2, d2 := dm.d1, d3 := dm.d3, d4 := dm.d4, d5 := dm.d5 }
  | 0, 2 => { d1 := dm.d3, d2 := dm.d2, d3 := dm.d1, d4 := dm.d4, d5 := dm.d5 }
  | 0, 3 => { d1 := dm.d4, d2 := dm.d2, d3 := dm.d3, d4 := dm.d1, d5 := dm.d5 }
  | 0, 4 => { d1 := dm.d5, d2 := dm.d2, d3 := dm.d3, d4 := dm.d4, d5 := dm.d1 }
  | 1, 2 => { d1 := dm.d1, d2 := dm.d3, d3 := dm.d2, d4 := dm.d4, d5 := dm.d5 }
  | 1, 3 => { d1 := dm.d1, d2 := dm.d4, d3 := dm.d3, d4 := dm.d2, d5 := dm.d5 }
  | 1, 4 => { d1 := dm.d1, d2 := dm.d5, d3 := dm.d3, d4 := dm.d4, d5 := dm.d2 }
  | 2, 3 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d4, d4 := dm.d3, d5 := dm.d5 }
  | 2, 4 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d5, d4 := dm.d4, d5 := dm.d3 }
  | 3, 4 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d3, d4 := dm.d5, d5 := dm.d4 }
  | _, _ => dm  -- For any other combination, return the original multiplication

theorem domino_swap_incorrect
  (dm : DominoMultiplication)
  (h : isCorrectMultiplication dm)
  (i j : Fin 5)
  (hne : i ≠ j) :
  ¬(isCorrectMultiplication (swapDominoes dm i j)) :=
by sorry

end domino_swap_incorrect_l425_42522


namespace lcm_812_smallest_lcm_812_24_smallest_lcm_812_672_l425_42591

theorem lcm_812_smallest (n : ℕ) : n > 0 ∧ Nat.lcm 812 n = 672 → n ≥ 24 := by
  sorry

theorem lcm_812_24 : Nat.lcm 812 24 = 672 := by
  sorry

theorem smallest_lcm_812_672 : ∃ (n : ℕ), n > 0 ∧ Nat.lcm 812 n = 672 ∧ ∀ (m : ℕ), m > 0 → Nat.lcm 812 m = 672 → m ≥ n := by
  sorry

end lcm_812_smallest_lcm_812_24_smallest_lcm_812_672_l425_42591


namespace subsets_with_adjacent_chairs_12_l425_42540

/-- The number of subsets with at least three adjacent chairs in a circular arrangement of 12 chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  let adjacent_3_to_6 := 4 * n
  let adjacent_7_plus := (Finset.range 6).sum (fun k => Nat.choose n (n - k))
  adjacent_3_to_6 + adjacent_7_plus

/-- Theorem stating that the number of subsets with at least three adjacent chairs
    in a circular arrangement of 12 chairs is 1634 -/
theorem subsets_with_adjacent_chairs_12 :
  subsets_with_adjacent_chairs 12 = 1634 := by
  sorry

end subsets_with_adjacent_chairs_12_l425_42540


namespace sum_in_base7_l425_42598

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 --/
def base10ToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The statement to prove --/
theorem sum_in_base7 :
  let a := [2, 1]  -- 12 in base 7
  let b := [5, 4, 2]  -- 245 in base 7
  let sum := base7ToBase10 a + base7ToBase10 b
  base10ToBase7 sum = [0, 6, 2] := by
  sorry

end sum_in_base7_l425_42598


namespace pythagorean_triple_identification_l425_42555

/-- A function that checks if three numbers form a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that (9, 12, 15) is the only Pythagorean triple among the given options -/
theorem pythagorean_triple_identification :
  (¬ is_pythagorean_triple 3 4 5) ∧
  (¬ is_pythagorean_triple 3 4 7) ∧
  (is_pythagorean_triple 9 12 15) :=
sorry

end pythagorean_triple_identification_l425_42555
