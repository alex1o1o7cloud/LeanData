import Mathlib

namespace equation_solution_l1218_121842

theorem equation_solution :
  ∃! x : ℚ, x ≠ -5 ∧ (x^2 + 3*x + 4) / (x + 5) = x + 7 :=
by
  use (-31 / 9)
  sorry

end equation_solution_l1218_121842


namespace one_bee_has_six_legs_l1218_121843

/-- The number of legs a bee has -/
def bee_legs : ℕ := sorry

/-- Two bees have 12 legs -/
axiom two_bees_legs : 2 * bee_legs = 12

/-- Prove that one bee has 6 legs -/
theorem one_bee_has_six_legs : bee_legs = 6 := by sorry

end one_bee_has_six_legs_l1218_121843


namespace charlies_share_l1218_121883

/-- Represents the share of money each person receives -/
structure Share where
  alice : ℚ
  bond : ℚ
  charlie : ℚ

/-- The conditions of the problem -/
def satisfiesConditions (s : Share) : Prop :=
  s.alice + s.bond + s.charlie = 1105 ∧
  (s.alice - 10) / (s.bond - 20) = 11 / 18 ∧
  (s.alice - 10) / (s.charlie - 15) = 11 / 24

/-- The theorem stating Charlie's share -/
theorem charlies_share :
  ∃ (s : Share), satisfiesConditions s ∧ s.charlie = 495 := by
  sorry


end charlies_share_l1218_121883


namespace inscribed_cylinder_properties_l1218_121887

/-- An equilateral cylinder inscribed in a regular tetrahedron --/
structure InscribedCylinder where
  a : ℝ  -- Edge length of the tetrahedron
  r : ℝ  -- Radius of the cylinder
  h : ℝ  -- Height of the cylinder
  cylinder_equilateral : h = 2 * r
  cylinder_inscribed : r = (a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6

/-- Theorem about the properties of the inscribed cylinder --/
theorem inscribed_cylinder_properties (c : InscribedCylinder) :
  c.r = (c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6 ∧
  (4 * Real.pi * c.r^2 : ℝ) = 4 * Real.pi * ((c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6)^2 ∧
  (2 * Real.pi * c.r^3 : ℝ) = 2 * Real.pi * ((c.a * (2 * Real.sqrt 3 - Real.sqrt 6)) / 6)^3 :=
by sorry

end inscribed_cylinder_properties_l1218_121887


namespace solve_for_a_l1218_121849

theorem solve_for_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x + 3 * y = 13) : a = 2 := by
  sorry

end solve_for_a_l1218_121849


namespace f_properties_l1218_121886

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x + 1) * Real.exp x

theorem f_properties :
  ∀ a : ℝ,
  (∃ x_min : ℝ, ∀ x : ℝ, f 0 x_min ≤ f 0 x ∧ f 0 x_min = -Real.exp (-2)) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ →
    (a < 0 → (x₂ < -2 ∨ x₂ > -1/a) → f a x₁ > f a x₂) ∧
    (a < 0 → -2 < x₁ ∧ x₂ < -1/a → f a x₁ < f a x₂) ∧
    (a = 0 → x₂ < -2 → f a x₁ > f a x₂) ∧
    (a = 0 → -2 < x₁ → f a x₁ < f a x₂) ∧
    (0 < a ∧ a < 1/2 → -1/a < x₁ ∧ x₂ < -2 → f a x₁ > f a x₂) ∧
    (0 < a ∧ a < 1/2 → (x₂ < -1/a ∨ -2 < x₁) → f a x₁ < f a x₂) ∧
    (a = 1/2 → f a x₁ < f a x₂) ∧
    (a > 1/2 → -2 < x₁ ∧ x₂ < -1/a → f a x₁ > f a x₂) ∧
    (a > 1/2 → (x₂ < -2 ∨ -1/a < x₁) → f a x₁ < f a x₂)) :=
by sorry

end f_properties_l1218_121886


namespace rectangle_ratio_l1218_121862

theorem rectangle_ratio (width : ℕ) (area : ℕ) (length : ℕ) : 
  width = 7 → 
  area = 196 → 
  length * width = area → 
  ∃ k : ℕ, length = k * width → 
  (length : ℚ) / width = 4 := by
sorry

end rectangle_ratio_l1218_121862


namespace specific_ellipse_foci_distance_l1218_121814

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (3, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 5 := by
  sorry

end specific_ellipse_foci_distance_l1218_121814


namespace volume_to_surface_area_ratio_is_one_fifth_l1218_121861

/-- Represents a shape created by joining nine unit cubes -/
structure CubeShape where
  /-- The total number of unit cubes in the shape -/
  total_cubes : ℕ
  /-- The number of exposed faces of the shape -/
  exposed_faces : ℕ
  /-- Assertion that the total number of cubes is 9 -/
  cube_count : total_cubes = 9
  /-- Assertion that the number of exposed faces is 45 -/
  face_count : exposed_faces = 45

/-- Calculates the ratio of volume to surface area for the cube shape -/
def volumeToSurfaceAreaRatio (shape : CubeShape) : ℚ :=
  shape.total_cubes / shape.exposed_faces

/-- Theorem stating that the ratio of volume to surface area is 1/5 -/
theorem volume_to_surface_area_ratio_is_one_fifth (shape : CubeShape) :
  volumeToSurfaceAreaRatio shape = 1 / 5 := by
  sorry

#check volume_to_surface_area_ratio_is_one_fifth

end volume_to_surface_area_ratio_is_one_fifth_l1218_121861


namespace inequality_equivalence_l1218_121806

theorem inequality_equivalence (x : ℝ) :
  (5 * x^2 + 20 * x - 34) / ((3 * x - 2) * (x - 5) * (x + 1)) < 2 ↔
  (-6 * x^3 + 27 * x^2 + 33 * x - 44) / ((3 * x - 2) * (x - 5) * (x + 1)) < 0 :=
by sorry

end inequality_equivalence_l1218_121806


namespace sum_always_positive_l1218_121817

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : is_monotone_increasing f)
  (h2 : is_odd_function f)
  (h3 : arithmetic_sequence a)
  (h4 : a 1 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end sum_always_positive_l1218_121817


namespace school_wall_stars_l1218_121826

theorem school_wall_stars (num_students : ℕ) (stars_per_student : ℕ) (total_stars : ℕ) :
  num_students = 210 →
  stars_per_student = 6 →
  total_stars = num_students * stars_per_student →
  total_stars = 1260 :=
by sorry

end school_wall_stars_l1218_121826


namespace range_of_a_l1218_121810

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x - a ≥ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ (Set.univ \ B a) → a > 2 := by
  sorry

end range_of_a_l1218_121810


namespace negation_of_universal_proposition_l1218_121827

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) := by
  sorry

end negation_of_universal_proposition_l1218_121827


namespace sets_inclusion_l1218_121804

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem sets_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end sets_inclusion_l1218_121804


namespace floor_times_self_eq_100_l1218_121825

theorem floor_times_self_eq_100 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 100 → x = 10 := by
  sorry

end floor_times_self_eq_100_l1218_121825


namespace time_after_1456_minutes_l1218_121808

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem time_after_1456_minutes :
  let start_time : Time := ⟨6, 0, by sorry⟩
  let elapsed_minutes : Nat := 1456
  let end_time : Time := addMinutes start_time elapsed_minutes
  end_time = ⟨6, 16, by sorry⟩ := by
  sorry

end time_after_1456_minutes_l1218_121808


namespace rogers_first_bag_l1218_121803

/-- Represents the number of candy bags a person has -/
def num_bags : ℕ := 2

/-- Represents the number of pieces in each of Sandra's bags -/
def sandra_pieces_per_bag : ℕ := 6

/-- Represents the number of pieces in Roger's second bag -/
def roger_second_bag : ℕ := 3

/-- Represents the difference in total pieces between Roger and Sandra -/
def roger_sandra_diff : ℕ := 2

/-- Represents the number of pieces in one of Roger's bags -/
def roger_one_bag : ℕ := 11

/-- Calculates the total number of candy pieces Sandra has -/
def sandra_total : ℕ := num_bags * sandra_pieces_per_bag

/-- Calculates the total number of candy pieces Roger has -/
def roger_total : ℕ := sandra_total + roger_sandra_diff

/-- Theorem: The number of pieces in Roger's first bag is 11 -/
theorem rogers_first_bag : roger_total - roger_second_bag = roger_one_bag :=
by sorry

end rogers_first_bag_l1218_121803


namespace complex_number_problem_l1218_121881

theorem complex_number_problem (z : ℂ) :
  Complex.abs z = 1 ∧ (Complex.I * Complex.im ((3 + 4*Complex.I) * z) = (3 + 4*Complex.I) * z) →
  z = 4/5 + 3/5*Complex.I ∨ z = -4/5 - 3/5*Complex.I := by
  sorry

end complex_number_problem_l1218_121881


namespace car_profit_theorem_l1218_121819

/-- Calculates the profit percentage on the original price of a car
    given the discount percentage on purchase and markup percentage on sale. -/
def profit_percentage (discount : ℝ) (markup : ℝ) : ℝ :=
  let purchase_price := 1 - discount
  let sale_price := purchase_price * (1 + markup)
  (sale_price - 1) * 100

/-- Theorem stating that buying a car at 5% discount and selling at 60% markup
    results in a 52% profit on the original price. -/
theorem car_profit_theorem :
  profit_percentage 0.05 0.60 = 52 := by sorry

end car_profit_theorem_l1218_121819


namespace regions_for_99_lines_l1218_121831

/-- The number of regions formed by a given number of lines in a plane -/
def num_regions (num_lines : ℕ) : Set ℕ :=
  {n | ∃ (configuration : Type) (f : configuration → ℕ), 
       (∀ c, f c ≤ (num_lines * (num_lines - 1)) / 2 + num_lines + 1) ∧
       (∃ c, f c = n)}

/-- Theorem stating that for 99 lines, the only possible numbers of regions less than 199 are 100 and 198 -/
theorem regions_for_99_lines :
  num_regions 99 ∩ {n | n < 199} = {100, 198} :=
by sorry

end regions_for_99_lines_l1218_121831


namespace total_spent_is_211_20_l1218_121816

/-- Calculates the total amount spent on a meal given the food price, sales tax rate, and tip rate. -/
def total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := food_price * (1 + sales_tax_rate)
  price_with_tax * (1 + tip_rate)

/-- Theorem stating that the total amount spent is $211.20 given the specified conditions. -/
theorem total_spent_is_211_20 :
  total_amount_spent 160 0.1 0.2 = 211.20 := by
  sorry

end total_spent_is_211_20_l1218_121816


namespace negative_integer_sum_and_square_is_fifteen_l1218_121854

theorem negative_integer_sum_and_square_is_fifteen (N : ℤ) : 
  N < 0 → N^2 + N = 15 → N = -5 := by sorry

end negative_integer_sum_and_square_is_fifteen_l1218_121854


namespace line_tangent_to_ellipse_l1218_121869

/-- The value of m^2 for which the line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 -/
theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 9 → 
    ∃! p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ p.2 = m * p.1 + 2) ↔ 
  m^2 = 1/3 :=
sorry

end line_tangent_to_ellipse_l1218_121869


namespace factorial_sum_perfect_square_l1218_121882

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumFactorials (m : ℕ) : ℕ := (List.range m).map factorial |>.sum

def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem factorial_sum_perfect_square :
  ∀ m : ℕ, m > 0 → (isPerfectSquare (sumFactorials m) ↔ m = 1 ∨ m = 3) :=
by sorry

end factorial_sum_perfect_square_l1218_121882


namespace shoe_ratio_proof_l1218_121812

theorem shoe_ratio_proof (total_shoes brown_shoes : ℕ) 
  (h1 : total_shoes = 66) 
  (h2 : brown_shoes = 22) : 
  (total_shoes - brown_shoes) / brown_shoes = 2 := by
sorry

end shoe_ratio_proof_l1218_121812


namespace parabola_and_intersection_properties_l1218_121800

/-- Parabola C with directrix x = -1/4 -/
def ParabolaC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = p.1}

/-- Line l passing through P(t, 0) -/
def LineL (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℝ, p.1 = m * p.2 + t}

/-- Points A and B are the intersections of ParabolaC and LineL -/
def IntersectionPoints (t : ℝ) : Set (ℝ × ℝ) :=
  ParabolaC ∩ LineL t

/-- Circle with diameter AB passes through the origin -/
def CircleThroughOrigin (t : ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, A ∈ IntersectionPoints t → B ∈ IntersectionPoints t →
    A.1 * B.1 + A.2 * B.2 = 0

theorem parabola_and_intersection_properties :
  (∀ p : ℝ × ℝ, p ∈ ParabolaC ↔ p.2^2 = p.1) ∧
  (∀ t : ℝ, CircleThroughOrigin t → t = 0 ∨ t = 1) :=
sorry

end parabola_and_intersection_properties_l1218_121800


namespace simplify_fraction_l1218_121891

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end simplify_fraction_l1218_121891


namespace product_of_powers_equals_fifty_l1218_121813

theorem product_of_powers_equals_fifty :
  (5^(2/10)) * (10^(4/10)) * (10^(1/10)) * (10^(5/10)) * (5^(8/10)) = 50 := by
  sorry

end product_of_powers_equals_fifty_l1218_121813


namespace product_of_sum_and_sum_of_cubes_l1218_121824

theorem product_of_sum_and_sum_of_cubes (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (sum_cubes_eq : x^3 + y^3 = 370) : 
  x * y = 21 := by
sorry

end product_of_sum_and_sum_of_cubes_l1218_121824


namespace cone_spheres_radius_theorem_l1218_121884

/-- A right circular cone with four congruent spheres inside --/
structure ConeWithSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : Bool
  spheres_count : Nat
  spheres_congruent : Bool
  spheres_tangent_to_each_other : Bool
  spheres_tangent_to_base : Bool
  spheres_tangent_to_side : Bool

/-- The theorem stating the relationship between cone dimensions and sphere radius --/
theorem cone_spheres_radius_theorem (c : ConeWithSpheres) :
  c.base_radius = 6 ∧
  c.height = 15 ∧
  c.is_right_circular = true ∧
  c.spheres_count = 4 ∧
  c.spheres_congruent = true ∧
  c.spheres_tangent_to_each_other = true ∧
  c.spheres_tangent_to_base = true ∧
  c.spheres_tangent_to_side = true →
  c.sphere_radius = 45 / 7 := by
sorry

end cone_spheres_radius_theorem_l1218_121884


namespace negation_equivalence_l1218_121892

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) := by sorry

end negation_equivalence_l1218_121892


namespace difference_ones_zeros_253_l1218_121818

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_ones (binary : List Bool) : ℕ :=
  sorry

def count_zeros (binary : List Bool) : ℕ :=
  sorry

theorem difference_ones_zeros_253 :
  let binary := binary_representation 253
  let ones := count_ones binary
  let zeros := count_zeros binary
  ones - zeros = 6 :=
sorry

end difference_ones_zeros_253_l1218_121818


namespace divisor_problem_l1218_121863

theorem divisor_problem (f y d : ℕ) : 
  (∃ k : ℕ, f = k * d + 3) →
  (∃ l : ℕ, y = l * d + 4) →
  (∃ m : ℕ, f + y = m * d + 2) →
  d = 5 := by
sorry

end divisor_problem_l1218_121863


namespace fifty_three_days_from_friday_l1218_121893

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

def days_in_week : Nat := 7

def friday_to_int : Nat := 5

def add_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match (friday_to_int + n) % days_in_week with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem fifty_three_days_from_friday :
  add_days DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry

end fifty_three_days_from_friday_l1218_121893


namespace quadratic_function_properties_l1218_121867

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 - 2

-- Theorem stating that f satisfies the given conditions
theorem quadratic_function_properties :
  (∃ (a : ℝ), f a = -2 ∧ ∀ x, f x ≥ f a) ∧  -- Vertex condition
  f 0 = 2                                   -- Y-intercept condition
  := by sorry

end quadratic_function_properties_l1218_121867


namespace stock_worth_l1218_121802

theorem stock_worth (X : ℝ) : 
  (0.1 * X * 1.2 + 0.9 * X * 0.95 = X - 400) → X = 16000 := by sorry

end stock_worth_l1218_121802


namespace mart_income_percentage_l1218_121859

def income_comparison (juan tim mart : ℝ) : Prop :=
  tim = juan * 0.6 ∧ mart = tim * 1.6

theorem mart_income_percentage (juan tim mart : ℝ) 
  (h : income_comparison juan tim mart) : mart = juan * 0.96 := by
  sorry

end mart_income_percentage_l1218_121859


namespace regular_octagon_extended_sides_angle_l1218_121878

/-- A regular octagon with vertices A, B, C, D, E, F, G, H -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- The angle formed by extending sides AB and GH of a regular octagon to meet at point Q -/
def angle_Q (octagon : RegularOctagon) : ℝ :=
  sorry

theorem regular_octagon_extended_sides_angle (octagon : RegularOctagon) :
  angle_Q octagon = 90 := by
  sorry

end regular_octagon_extended_sides_angle_l1218_121878


namespace complex_equation_solution_l1218_121822

theorem complex_equation_solution :
  ∀ a : ℂ, (1 - I)^3 / (1 + I) = a + 3*I → a = -2 := by
  sorry

end complex_equation_solution_l1218_121822


namespace book_sale_problem_l1218_121860

theorem book_sale_problem (cost_loss : ℝ) (sale_price : ℝ) :
  cost_loss = 315 →
  sale_price = cost_loss * 0.85 →
  sale_price = (cost_loss + (2565 - 315)) * 1.19 →
  cost_loss + (2565 - 315) = 2565 := by
  sorry

end book_sale_problem_l1218_121860


namespace triangle_is_right_angled_l1218_121880

theorem triangle_is_right_angled : 
  let A : ℂ := 1
  let B : ℂ := Complex.I * 2
  let C : ℂ := 5 + Complex.I * 2
  let AB : ℂ := B - A
  let BC : ℂ := C - B
  let CA : ℂ := A - C
  Complex.abs AB ^ 2 + Complex.abs CA ^ 2 = Complex.abs BC ^ 2 := by
  sorry

end triangle_is_right_angled_l1218_121880


namespace minimum_canvas_dimensions_l1218_121836

/-- Represents the dimensions of a canvas --/
structure CanvasDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle given its width and height --/
def rectangleArea (w h : ℝ) : ℝ := w * h

/-- Represents the constraints for the canvas problem --/
structure CanvasConstraints where
  miniatureArea : ℝ
  topBottomMargin : ℝ
  sideMargin : ℝ

/-- Calculates the total canvas dimensions given the miniature dimensions and margins --/
def totalCanvasDimensions (miniWidth miniHeight topBottomMargin sideMargin : ℝ) : CanvasDimensions :=
  { width := miniWidth + 2 * sideMargin,
    height := miniHeight + 2 * topBottomMargin }

/-- Theorem stating the minimum dimensions of the required canvas --/
theorem minimum_canvas_dimensions (constraints : CanvasConstraints) 
  (h1 : constraints.miniatureArea = 72)
  (h2 : constraints.topBottomMargin = 4)
  (h3 : constraints.sideMargin = 2) :
  ∃ (minCanvas : CanvasDimensions),
    minCanvas.width = 10 ∧ 
    minCanvas.height = 20 ∧ 
    ∀ (canvas : CanvasDimensions),
      (∃ (miniWidth miniHeight : ℝ),
        rectangleArea miniWidth miniHeight = constraints.miniatureArea ∧
        canvas = totalCanvasDimensions miniWidth miniHeight constraints.topBottomMargin constraints.sideMargin) →
      canvas.width * canvas.height ≥ minCanvas.width * minCanvas.height :=
sorry

end minimum_canvas_dimensions_l1218_121836


namespace rotation_composition_l1218_121851

/-- Represents a rotation in a plane -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Represents a translation in a plane -/
structure Translation where
  direction : ℝ × ℝ

/-- Represents the result of composing two rotations -/
inductive RotationComposition
  | IsRotation : Rotation → RotationComposition
  | IsTranslation : Translation → RotationComposition

/-- 
  Theorem: The composition of two rotations is either a rotation or a translation
  depending on the sum of their angles.
-/
theorem rotation_composition (r1 r2 : Rotation) :
  ∃ (result : RotationComposition),
    (¬ ∃ (k : ℤ), r1.angle + r2.angle = 2 * π * k → 
      ∃ (c : ℝ × ℝ), result = RotationComposition.IsRotation ⟨c, r1.angle + r2.angle⟩) ∧
    (∃ (k : ℤ), r1.angle + r2.angle = 2 * π * k → 
      ∃ (d : ℝ × ℝ), result = RotationComposition.IsTranslation ⟨d⟩) :=
by sorry


end rotation_composition_l1218_121851


namespace mans_speed_in_still_water_l1218_121844

/-- 
Given a man's upstream and downstream rowing speeds, 
calculate his speed in still water.
-/
theorem mans_speed_in_still_water 
  (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 20) 
  (h2 : downstream_speed = 60) : 
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

#check mans_speed_in_still_water

end mans_speed_in_still_water_l1218_121844


namespace triangle_area_sqrt_3_l1218_121839

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that its area is √3 -/
theorem triangle_area_sqrt_3 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b * Real.cos C + c * Real.cos B = a * Real.cos C + c * Real.cos A)
  (h2 : b * Real.cos C + c * Real.cos B = 2)
  (h3 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c) :
  (1/2) * a * b * Real.sin C = Real.sqrt 3 := by
  sorry

end triangle_area_sqrt_3_l1218_121839


namespace cereal_eating_time_l1218_121838

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate : ℚ) (thin_rate : ℚ) (amount : ℚ) : ℚ :=
  amount / (fat_rate + thin_rate)

/-- Theorem: Given the eating rates and amount of cereal, prove that it takes 45 minutes to finish -/
theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 45  -- Mr. Thin's eating rate in pounds per minute
  let amount : ℚ := 4  -- Amount of cereal in pounds
  time_to_eat_together fat_rate thin_rate amount = 45 := by
  sorry

#eval time_to_eat_together (1/15 : ℚ) (1/45 : ℚ) 4

end cereal_eating_time_l1218_121838


namespace two_std_dev_below_mean_l1218_121888

def normal_distribution (μ σ : ℝ) : Type := sorry

theorem two_std_dev_below_mean 
  (μ σ : ℝ) 
  (dist : normal_distribution μ σ) 
  (h_μ : μ = 14.5) 
  (h_σ : σ = 1.5) : 
  μ - 2 * σ = 11.5 := by
  sorry

end two_std_dev_below_mean_l1218_121888


namespace papi_calot_plants_to_buy_l1218_121846

/-- Calculates the total number of plants needed for a given crop -/
def totalPlants (rows : ℕ) (plantsPerRow : ℕ) (additional : ℕ) : ℕ :=
  rows * plantsPerRow + additional

/-- Represents Papi Calot's garden planning -/
structure GardenPlan where
  potatoRows : ℕ
  potatoPlantsPerRow : ℕ
  additionalPotatoes : ℕ
  carrotRows : ℕ
  carrotPlantsPerRow : ℕ
  additionalCarrots : ℕ
  onionRows : ℕ
  onionPlantsPerRow : ℕ
  additionalOnions : ℕ

/-- Theorem stating the correct number of plants Papi Calot needs to buy -/
theorem papi_calot_plants_to_buy (plan : GardenPlan)
  (h_potato : plan.potatoRows = 10 ∧ plan.potatoPlantsPerRow = 25 ∧ plan.additionalPotatoes = 20)
  (h_carrot : plan.carrotRows = 15 ∧ plan.carrotPlantsPerRow = 30 ∧ plan.additionalCarrots = 30)
  (h_onion : plan.onionRows = 12 ∧ plan.onionPlantsPerRow = 20 ∧ plan.additionalOnions = 10) :
  totalPlants plan.potatoRows plan.potatoPlantsPerRow plan.additionalPotatoes = 270 ∧
  totalPlants plan.carrotRows plan.carrotPlantsPerRow plan.additionalCarrots = 480 ∧
  totalPlants plan.onionRows plan.onionPlantsPerRow plan.additionalOnions = 250 := by
  sorry

end papi_calot_plants_to_buy_l1218_121846


namespace intersection_of_lines_l1218_121835

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (7/18, -1/6)

/-- First line equation: y = -3x + 1 -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- Second line equation: y + 4 = 15x - 2 -/
def line2 (x y : ℚ) : Prop := y + 4 = 15 * x - 2

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end intersection_of_lines_l1218_121835


namespace tom_age_l1218_121841

theorem tom_age (carla_age : ℕ) (tom_age : ℕ) (dave_age : ℕ) : 
  (tom_age = 2 * carla_age - 1) →
  (dave_age = carla_age + 3) →
  (carla_age + tom_age + dave_age = 30) →
  tom_age = 13 := by
sorry

end tom_age_l1218_121841


namespace total_basketballs_donated_prove_total_basketballs_l1218_121872

/-- Calculates the total number of basketballs donated to a school --/
theorem total_basketballs_donated (total_donations : ℕ) (basketball_hoops : ℕ) (pool_floats : ℕ) 
  (footballs : ℕ) (tennis_balls : ℕ) : ℕ :=
  let basketballs_with_hoops := basketball_hoops / 2
  let undamaged_pool_floats := pool_floats * 3 / 4
  let accounted_donations := basketball_hoops + undamaged_pool_floats + footballs + tennis_balls
  let separate_basketballs := total_donations - accounted_donations
  basketballs_with_hoops + separate_basketballs

/-- Proves that the total number of basketballs donated is 90 --/
theorem prove_total_basketballs :
  total_basketballs_donated 300 60 120 50 40 = 90 := by
  sorry

end total_basketballs_donated_prove_total_basketballs_l1218_121872


namespace second_number_proof_l1218_121845

theorem second_number_proof (N : ℕ) : 
  (N % 144 = 29) → (6215 % 144 = 23) → N = 6365 :=
by
  sorry

end second_number_proof_l1218_121845


namespace half_angle_quadrants_l1218_121896

theorem half_angle_quadrants (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) →
  (∃ n : ℤ, 2 * n * π < α / 2 ∧ α / 2 < 2 * n * π + π / 2) ∨
  (∃ n : ℤ, (2 * n + 1) * π < α / 2 ∧ α / 2 < (2 * n + 1) * π + π / 2) := by
sorry


end half_angle_quadrants_l1218_121896


namespace find_b_l1218_121837

theorem find_b (a b c : ℚ) 
  (sum_eq : a + b + c = 150)
  (eq_after_changes : a + 10 = b - 5 ∧ b - 5 = 7 * c) : 
  b = 232 / 3 := by
sorry

end find_b_l1218_121837


namespace geometric_sequence_ratio_l1218_121856

/-- Given a geometric sequence with four terms and common ratio 2,
    prove that (2a₁ + a₂) / (2a₃ + a₄) = 1/4 -/
theorem geometric_sequence_ratio (a₁ a₂ a₃ a₄ : ℝ) :
  a₂ = 2 * a₁ → a₃ = 2 * a₂ → a₄ = 2 * a₃ →
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := by
sorry

end geometric_sequence_ratio_l1218_121856


namespace polynomial_division_l1218_121876

theorem polynomial_division (a b : ℝ) (h : b ≠ 2 * a) :
  (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b := by
  sorry

end polynomial_division_l1218_121876


namespace existence_of_a_for_minimum_value_l1218_121858

theorem existence_of_a_for_minimum_value (e : Real) (h_e : e > 0) : ∃ a : Real,
  (∀ x : Real, 0 < x ∧ x ≤ e → ax - Real.log x ≥ 3) ∧
  (∃ x : Real, 0 < x ∧ x ≤ e ∧ ax - Real.log x = 3) ∧
  a = Real.exp 2 := by
  sorry

end existence_of_a_for_minimum_value_l1218_121858


namespace intersection_line_of_circles_l1218_121890

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line equation passing through the intersection points of two circles --/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 23/8

theorem intersection_line_of_circles :
  let c1 : Circle := { center := (0, 0), radius := 5 }
  let c2 : Circle := { center := (4, 4), radius := 3 }
  ∀ x y : ℝ, (x^2 + y^2 = c1.radius^2) ∧ ((x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2) →
    intersectionLine c1 c2 x y := by
  sorry

end intersection_line_of_circles_l1218_121890


namespace repeating_decimal_division_l1218_121895

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.repeatingPart / (99 : ℚ)

/-- The repeating decimal 0.72̅ -/
def zero_point_72_repeating : RepeatingDecimal :=
  ⟨0, 72⟩

/-- The repeating decimal 2.09̅ -/
def two_point_09_repeating : RepeatingDecimal :=
  ⟨2, 9⟩

/-- Theorem stating that the division of the two given repeating decimals equals 8/23 -/
theorem repeating_decimal_division :
    (toRational zero_point_72_repeating) / (toRational two_point_09_repeating) = 8 / 23 := by
  sorry


end repeating_decimal_division_l1218_121895


namespace raft_existence_l1218_121823

-- Define the river shape
def RiverShape : Type := sorry

-- Define the path of the chip
def ChipPath (river : RiverShape) : Type := sorry

-- Define the raft shape
def RaftShape : Type := sorry

-- Function to check if a raft touches both banks at all points
def touchesBothBanks (river : RiverShape) (raft : RaftShape) (path : ChipPath river) : Prop := sorry

-- Theorem statement
theorem raft_existence (river : RiverShape) (chip_path : ChipPath river) :
  ∃ (raft : RaftShape), touchesBothBanks river raft chip_path := by
  sorry

end raft_existence_l1218_121823


namespace inequality_proof_l1218_121840

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) :=
by sorry

end inequality_proof_l1218_121840


namespace customers_added_during_lunch_rush_l1218_121834

theorem customers_added_during_lunch_rush 
  (initial_customers : ℕ) 
  (no_tip_customers : ℕ) 
  (tip_customers : ℕ) 
  (h1 : initial_customers = 29)
  (h2 : no_tip_customers = 34)
  (h3 : tip_customers = 15)
  (h4 : no_tip_customers + tip_customers = initial_customers + (customers_added : ℕ)) :
  customers_added = 20 :=
by sorry

end customers_added_during_lunch_rush_l1218_121834


namespace expression_is_integer_l1218_121809

theorem expression_is_integer (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  ∃ k : ℤ, k = (x^n / ((x-y)*(x-z))) + (y^n / ((y-x)*(y-z))) + (z^n / ((z-x)*(z-y))) :=
by sorry

end expression_is_integer_l1218_121809


namespace angle_C_in_triangle_ABC_l1218_121899

theorem angle_C_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) : 
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  C = π / 6  -- 30° in radians
:= by sorry

end angle_C_in_triangle_ABC_l1218_121899


namespace det_E_equals_25_l1218_121801

/-- A 2x2 matrix representing a dilation by factor 5 centered at the origin -/
def D : Matrix (Fin 2) (Fin 2) ℝ := !![5, 0; 0, 5]

/-- A 2x2 matrix representing a 90-degree counter-clockwise rotation -/
def R : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The combined transformation matrix E -/
def E : Matrix (Fin 2) (Fin 2) ℝ := R * D

theorem det_E_equals_25 : Matrix.det E = 25 := by
  sorry

end det_E_equals_25_l1218_121801


namespace rectangle_area_l1218_121821

/-- The area of a rectangle with width 10 meters and length 2 meters is 20 square meters. -/
theorem rectangle_area : 
  ∀ (width length area : ℝ), 
  width = 10 → 
  length = 2 → 
  area = width * length → 
  area = 20 := by
sorry

end rectangle_area_l1218_121821


namespace sector_area_l1218_121889

theorem sector_area (θ : Real) (r : Real) (h1 : θ = 135) (h2 : r = 20) :
  (θ * π * r^2) / 360 = 150 * π := by
  sorry

end sector_area_l1218_121889


namespace tan_neg_x_domain_l1218_121830

theorem tan_neg_x_domain :
  {x : ℝ | ∀ n : ℤ, x ≠ -π/2 + n*π} = {x : ℝ | ∃ y : ℝ, y = Real.tan (-x)} :=
by sorry

end tan_neg_x_domain_l1218_121830


namespace problem_statement_l1218_121853

theorem problem_statement (x y : ℝ) 
  (h1 : 4 + x = 5 - y) 
  (h2 : 3 + y = 6 + x) : 
  4 - x = 5 := by
sorry

end problem_statement_l1218_121853


namespace machine_sale_price_l1218_121847

def selling_price (purchase_price repair_cost transport_cost profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

theorem machine_sale_price :
  selling_price 11000 5000 1000 50 = 25500 := by
  sorry

end machine_sale_price_l1218_121847


namespace min_c_value_l1218_121894

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c) :
  (∃! x y : ℝ, 2 * x + y = 2031 ∧ y = |x - a| + |x - b| + |x - c|) →
  c ≥ 1016 ∧ ∃ a' b' : ℕ, a' < b' ∧ b' < 1016 ∧
    (∃! x y : ℝ, 2 * x + y = 2031 ∧ y = |x - a'| + |x - b'| + |x - 1016|) :=
by sorry

end min_c_value_l1218_121894


namespace chord_length_specific_case_l1218_121877

/-- The length of the chord formed by the intersection of a line and a circle -/
def chord_length (line_point : ℝ × ℝ) (line_angle : ℝ) (circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  sorry

theorem chord_length_specific_case :
  let line_point : ℝ × ℝ := (1, 0)
  let line_angle : ℝ := 30 * π / 180  -- 30 degrees in radians
  let circle_center : ℝ × ℝ := (2, 0)
  let circle_radius : ℝ := 1
  chord_length line_point line_angle circle_center circle_radius = Real.sqrt 3 := by
  sorry

end chord_length_specific_case_l1218_121877


namespace marble_selection_theorem_l1218_121864

/-- The number of marbles John has in total -/
def total_marbles : ℕ := 15

/-- The number of colors with exactly two marbles each -/
def special_colors : ℕ := 3

/-- The number of marbles for each special color -/
def marbles_per_special_color : ℕ := 2

/-- The number of marbles to be chosen -/
def marbles_to_choose : ℕ := 5

/-- The number of special colored marbles to be chosen -/
def special_marbles_to_choose : ℕ := 2

/-- The number of ways to choose the marbles under the given conditions -/
def ways_to_choose : ℕ := 1008

theorem marble_selection_theorem :
  (Nat.choose special_colors special_marbles_to_choose) *
  (Nat.choose marbles_per_special_color 1) ^ special_marbles_to_choose *
  (Nat.choose (total_marbles - special_colors * marbles_per_special_color) (marbles_to_choose - special_marbles_to_choose)) =
  ways_to_choose := by
  sorry

end marble_selection_theorem_l1218_121864


namespace gcd_of_f_over_primes_ge_11_l1218_121871

-- Define the function f(p)
def f (p : ℕ) : ℕ := p^6 - 7*p^2 + 6

-- Define the set of prime numbers greater than or equal to 11
def P : Set ℕ := {p : ℕ | Nat.Prime p ∧ p ≥ 11}

-- Theorem statement
theorem gcd_of_f_over_primes_ge_11 : 
  ∃ (d : ℕ), d > 0 ∧ (∀ (p : ℕ), p ∈ P → (f p).gcd d = d) ∧ 
  (∀ (m : ℕ), (∀ (p : ℕ), p ∈ P → (f p).gcd m = m) → m ≤ d) ∧ d = 16 := by
sorry

end gcd_of_f_over_primes_ge_11_l1218_121871


namespace parabola_line_intersection_l1218_121811

/-- The parabola function -/
def f (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- The line function -/
def g (k : ℝ) : ℝ → ℝ := λ _ ↦ k

/-- The condition for a single intersection point -/
def has_single_intersection (k : ℝ) : Prop :=
  ∃! y, f y = g k y

theorem parabola_line_intersection :
  ∀ k, has_single_intersection k ↔ k = 13/3 := by sorry

end parabola_line_intersection_l1218_121811


namespace jessie_points_l1218_121807

def total_points : ℕ := 311
def some_players_points : ℕ := 188
def num_equal_scorers : ℕ := 3

theorem jessie_points : 
  (total_points - some_players_points) / num_equal_scorers = 41 := by
  sorry

end jessie_points_l1218_121807


namespace handshakes_and_highfives_l1218_121897

/-- The number of unique pairings in a group of n people -/
def uniquePairings (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of people at the gathering -/
def numberOfPeople : ℕ := 12

theorem handshakes_and_highfives :
  uniquePairings numberOfPeople = 66 ∧
  uniquePairings numberOfPeople = 66 := by
  sorry

#eval uniquePairings numberOfPeople

end handshakes_and_highfives_l1218_121897


namespace average_equals_median_l1218_121828

theorem average_equals_median (n : ℕ) (k : ℕ) (x : ℝ) : 
  n > 0 → 
  k > 0 → 
  x > 0 → 
  n = 14 → 
  (x * (k + 1) / 2)^2 = (2 * n)^2 → 
  x = n := by
sorry

end average_equals_median_l1218_121828


namespace hoopit_toes_count_l1218_121875

/-- Represents the number of toes a Hoopit has on each hand -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands a Hoopit has -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes a Neglart has on each hand -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands a Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

theorem hoopit_toes_count : 
  hoopit_toes_per_hand * hoopit_hands * hoopit_students + 
  neglart_toes_per_hand * neglart_hands * neglart_students = total_toes :=
by sorry

end hoopit_toes_count_l1218_121875


namespace apple_basket_problem_l1218_121885

/-- The number of baskets in the apple-picking problem -/
def number_of_baskets : ℕ := 11

/-- The total number of apples initially -/
def total_apples : ℕ := 1000

/-- The number of apples left after picking -/
def apples_left : ℕ := 340

/-- The number of children picking apples -/
def number_of_children : ℕ := 10

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem apple_basket_problem :
  (number_of_children * sum_of_first_n number_of_baskets = total_apples - apples_left) ∧
  (number_of_baskets > 0) :=
sorry

end apple_basket_problem_l1218_121885


namespace angle_complement_relation_l1218_121855

theorem angle_complement_relation (x : ℝ) : x = 70 → x = 2 * (90 - x) + 30 := by
  sorry

end angle_complement_relation_l1218_121855


namespace find_M_l1218_121866

theorem find_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1800) ∧ (M = 2520) := by
  sorry

end find_M_l1218_121866


namespace inequality_range_l1218_121865

theorem inequality_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) 
  ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) :=
sorry

end inequality_range_l1218_121865


namespace losing_teams_total_score_l1218_121833

/-- Represents a basketball game between two teams -/
structure Game where
  team1_score : ℕ
  team2_score : ℕ

/-- The total score of a game -/
def Game.total_score (g : Game) : ℕ := g.team1_score + g.team2_score

/-- The margin of victory in a game -/
def Game.margin (g : Game) : ℤ := g.team1_score - g.team2_score

theorem losing_teams_total_score (game1 game2 : Game) 
  (h1 : game1.total_score = 150)
  (h2 : game1.margin = 10)
  (h3 : game2.total_score = 140)
  (h4 : game2.margin = -20) :
  game1.team2_score + game2.team1_score = 130 := by
sorry

end losing_teams_total_score_l1218_121833


namespace ten_percent_increase_l1218_121832

theorem ten_percent_increase (original : ℝ) (increased : ℝ) : 
  original = 600 → increased = original * 1.1 → increased = 660 := by
  sorry

end ten_percent_increase_l1218_121832


namespace functional_equation_solution_l1218_121805

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, g (x^2 + y^2 + y * g z) = x * g x + z^2 * g y

/-- The theorem stating that g must be either the zero function or the identity function -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
    (∀ x, g x = 0) ∨ (∀ x, g x = x) := by
  sorry

end functional_equation_solution_l1218_121805


namespace evaluate_expression_l1218_121857

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 := by
  sorry

end evaluate_expression_l1218_121857


namespace sun_division_problem_l1218_121820

/-- Prove that the total amount is 105 given the conditions of the sun division problem -/
theorem sun_division_problem (x y z : ℝ) : 
  (y = 0.45 * x) →  -- For each rupee x gets, y gets 45 paisa
  (z = 0.30 * x) →  -- For each rupee x gets, z gets 30 paisa
  (y = 27) →        -- y's share is Rs. 27
  (x + y + z = 105) -- The total amount is 105
  := by sorry

end sun_division_problem_l1218_121820


namespace equation_one_solutions_equation_two_solution_l1218_121848

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  4 * (x + 1)^2 - 25 = 0 ↔ x = 3/2 ∨ x = -7/2 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 10)^3 = -125 ↔ x = -15 := by sorry

end equation_one_solutions_equation_two_solution_l1218_121848


namespace sons_age_l1218_121850

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end sons_age_l1218_121850


namespace no_quadratic_factorization_l1218_121874

theorem no_quadratic_factorization :
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end no_quadratic_factorization_l1218_121874


namespace nickys_card_value_l1218_121852

/-- Proves that if Nicky trades two cards of equal value for one card worth $21 
    and makes a profit of $5, then each of Nicky's cards is worth $8. -/
theorem nickys_card_value (card_value : ℝ) : 
  (2 * card_value + 5 = 21) → card_value = 8 := by
  sorry

end nickys_card_value_l1218_121852


namespace factorization_a_squared_minus_3a_l1218_121829

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3*a = a*(a - 3) := by
  sorry

end factorization_a_squared_minus_3a_l1218_121829


namespace simplify_fraction_l1218_121870

theorem simplify_fraction : (150 : ℚ) / 6000 * 75 = 15 / 8 := by
  sorry

end simplify_fraction_l1218_121870


namespace smallest_cube_for_cone_l1218_121879

/-- Represents a cone with given height and base diameter -/
structure Cone where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- The volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- A cube contains a cone if its side length is at least as large as both
    the cone's height and base diameter -/
def cubeContainsCone (cube : Cube) (cone : Cone) : Prop :=
  cube.sideLength ≥ cone.height ∧ cube.sideLength ≥ cone.baseDiameter

theorem smallest_cube_for_cone (c : Cone)
    (h1 : c.height = 15)
    (h2 : c.baseDiameter = 8) :
    ∃ (cube : Cube),
      cubeContainsCone cube c ∧
      cubeVolume cube = 3375 ∧
      ∀ (other : Cube), cubeContainsCone other c → cubeVolume other ≥ cubeVolume cube :=
  sorry

end smallest_cube_for_cone_l1218_121879


namespace factorization_equality_l1218_121873

theorem factorization_equality (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) := by
  sorry

end factorization_equality_l1218_121873


namespace squeak_interval_is_nine_seconds_l1218_121815

/-- Represents a gear mechanism with two gears -/
structure GearMechanism where
  small_gear_teeth : ℕ
  large_gear_teeth : ℕ
  large_gear_revolution_time : ℝ

/-- Calculates the time interval between squeaks for a gear mechanism -/
def squeak_interval (gm : GearMechanism) : ℝ :=
  let lcm := Nat.lcm gm.small_gear_teeth gm.large_gear_teeth
  let large_gear_revolutions := lcm / gm.large_gear_teeth
  large_gear_revolutions * gm.large_gear_revolution_time

/-- Theorem stating that for the given gear mechanism, the squeak interval is 9 seconds -/
theorem squeak_interval_is_nine_seconds (gm : GearMechanism) 
  (h1 : gm.small_gear_teeth = 12) 
  (h2 : gm.large_gear_teeth = 32) 
  (h3 : gm.large_gear_revolution_time = 3) : 
  squeak_interval gm = 9 := by
  sorry

#eval squeak_interval { small_gear_teeth := 12, large_gear_teeth := 32, large_gear_revolution_time := 3 }

end squeak_interval_is_nine_seconds_l1218_121815


namespace intersection_in_fourth_quadrant_implies_m_range_l1218_121898

theorem intersection_in_fourth_quadrant_implies_m_range 
  (m : ℝ) 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x + y - 3*m = 0)
  (h2 : ∀ x y, line2 x y ↔ 2*x - y + 2*m - 1 = 0)
  (h_intersect : ∃ x y, line1 x y ∧ line2 x y ∧ x > 0 ∧ y < 0) :
  -1 < m ∧ m < 1/8 := by
sorry

end intersection_in_fourth_quadrant_implies_m_range_l1218_121898


namespace a2_value_l1218_121868

theorem a2_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 = 
    a₀ + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + 
    a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7) →
  a₂ = 56 := by
sorry

end a2_value_l1218_121868
