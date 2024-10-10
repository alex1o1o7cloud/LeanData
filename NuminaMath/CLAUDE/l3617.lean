import Mathlib

namespace ellipse_b_value_l3617_361708

/-- An ellipse with foci at (1, 1) and (1, -1) passing through (7, 0) -/
structure Ellipse where
  foci1 : ℝ × ℝ := (1, 1)
  foci2 : ℝ × ℝ := (1, -1)
  point : ℝ × ℝ := (7, 0)

/-- The standard form of an ellipse equation -/
def standard_equation (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem stating that b = 6 for the given ellipse -/
theorem ellipse_b_value (e : Ellipse) :
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧
  standard_equation h k a b (e.point.1) (e.point.2) ∧
  b = 6 := by
  sorry

end ellipse_b_value_l3617_361708


namespace total_teachers_l3617_361729

theorem total_teachers (senior : ℕ) (intermediate : ℕ) (sampled_total : ℕ) (sampled_other : ℕ)
  (h1 : senior = 26)
  (h2 : intermediate = 104)
  (h3 : sampled_total = 56)
  (h4 : sampled_other = 16)
  (h5 : ∀ (category : ℕ) (sampled_category : ℕ) (total : ℕ),
    (category : ℚ) / total = (sampled_category : ℚ) / sampled_total) :
  ∃ (total : ℕ), total = 52 := by
sorry

end total_teachers_l3617_361729


namespace cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six_square_minus_product_equals_five_plus_two_sqrt_three_l3617_361794

-- Part 1
theorem cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six :
  (8 : ℝ) ^ (1/3) - (Real.sqrt 12 * Real.sqrt 6) / Real.sqrt 3 = 2 - 2 * Real.sqrt 6 := by sorry

-- Part 2
theorem square_minus_product_equals_five_plus_two_sqrt_three :
  (Real.sqrt 3 + 1)^2 - (2 * Real.sqrt 2 + 3) * (2 * Real.sqrt 2 - 3) = 5 + 2 * Real.sqrt 3 := by sorry

end cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six_square_minus_product_equals_five_plus_two_sqrt_three_l3617_361794


namespace smallest_dual_base_representation_l3617_361762

/-- Represents a number in a given base with repeated digits -/
def repeatedDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid for a given base -/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (A C : Nat),
    isValidDigit A 8 ∧
    isValidDigit C 6 ∧
    repeatedDigitNumber A 8 = repeatedDigitNumber C 6 ∧
    repeatedDigitNumber A 8 = 19 ∧
    (∀ (A' C' : Nat),
      isValidDigit A' 8 →
      isValidDigit C' 6 →
      repeatedDigitNumber A' 8 = repeatedDigitNumber C' 6 →
      repeatedDigitNumber A' 8 ≥ 19) :=
by
  sorry

end smallest_dual_base_representation_l3617_361762


namespace average_sale_is_3500_l3617_361745

def sales : List ℕ := [3435, 3920, 3855, 4230, 3560, 2000]

theorem average_sale_is_3500 : 
  (sales.sum / sales.length : ℚ) = 3500 := by
  sorry

end average_sale_is_3500_l3617_361745


namespace milk_water_ratio_l3617_361714

theorem milk_water_ratio 
  (initial_volume : ℝ) 
  (initial_milk_ratio : ℝ) 
  (initial_water_ratio : ℝ) 
  (added_water : ℝ) : 
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 21 → 
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let new_water := initial_water + added_water
  let new_ratio_milk := initial_milk / (initial_milk + new_water) * 11
  let new_ratio_water := new_water / (initial_milk + new_water) * 11
  new_ratio_milk = 6 ∧ new_ratio_water = 5 :=
by sorry


end milk_water_ratio_l3617_361714


namespace special_triangle_exists_l3617_361700

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorFunction : Point → Color := sorry

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the circumradius of a triangle
def circumradius (t : Triangle) : ℝ := sorry

-- Define a predicate for monochromatic triangle
def isMonochromatic (t : Triangle) : Prop :=
  colorFunction t.A = colorFunction t.B ∧ colorFunction t.B = colorFunction t.C

-- Define a predicate for angle ratio condition
def satisfiesAngleRatio (t : Triangle) : Prop := sorry

-- The main theorem
theorem special_triangle_exists :
  ∃ (t : Triangle), isMonochromatic t ∧ circumradius t = 2008 ∧ satisfiesAngleRatio t := by
  sorry

end special_triangle_exists_l3617_361700


namespace projection_coplanarity_l3617_361797

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with a quadrilateral base -/
structure Pyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D

/-- Checks if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersectionPoint (p1 p2 p3 p4 : Point3D) : Point3D := sorry

/-- Checks if a point is the height of a pyramid -/
def isHeight (p : Point3D) (pyr : Pyramid) : Prop := sorry

/-- Projects a point onto a plane defined by three points -/
def projectOntoPlane (p : Point3D) (p1 p2 p3 : Point3D) : Point3D := sorry

/-- Checks if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

theorem projection_coplanarity (pyr : Pyramid) : 
  let M := intersectionPoint pyr.A pyr.C pyr.B pyr.D
  isPerpendicular pyr.A pyr.C pyr.B pyr.D ∧ 
  isHeight (intersectionPoint pyr.E M pyr.A pyr.C) pyr →
  areCoplanar 
    (projectOntoPlane M pyr.E pyr.A pyr.B)
    (projectOntoPlane M pyr.E pyr.B pyr.C)
    (projectOntoPlane M pyr.E pyr.C pyr.D)
    (projectOntoPlane M pyr.E pyr.D pyr.A) := by
  sorry

end projection_coplanarity_l3617_361797


namespace cricket_bat_profit_percentage_l3617_361773

/-- The profit percentage of a cricket bat sale -/
def profit_percentage (selling_price profit : ℚ) : ℚ :=
  (profit / (selling_price - profit)) * 100

/-- Theorem: The profit percentage is 36% when a cricket bat is sold for $850 with a profit of $225 -/
theorem cricket_bat_profit_percentage :
  profit_percentage 850 225 = 36 := by
  sorry

end cricket_bat_profit_percentage_l3617_361773


namespace trig_identity_l3617_361783

theorem trig_identity (α : ℝ) : Real.sin α ^ 6 + Real.cos α ^ 6 + 3 * Real.sin α ^ 2 * Real.cos α ^ 2 = 1 := by
  sorry

end trig_identity_l3617_361783


namespace orthogonal_medians_theorem_l3617_361775

/-- Given a triangle with side lengths a, b, c and medians ma, mb, mc,
    if ma is perpendicular to mb, then the medians form a right-angled triangle
    and the inequality 5(a^2 + b^2 - c^2) ≥ 8ab holds. -/
theorem orthogonal_medians_theorem (a b c ma mb mc : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_mb : mb^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_mc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4)
  (h_perp : ma * mb = 0) : 
  ma^2 + mb^2 = mc^2 ∧ 5*(a^2 + b^2 - c^2) ≥ 8*a*b :=
sorry

end orthogonal_medians_theorem_l3617_361775


namespace circle_centers_line_l3617_361709

/-- The set of circles C_k defined by (x-k+1)^2 + (y-3k)^2 = 2k^4 where k is a positive integer -/
def CircleSet (k : ℕ+) (x y : ℝ) : Prop :=
  (x - k + 1)^2 + (y - 3*k)^2 = 2 * k^4

/-- The center of circle C_k -/
def CircleCenter (k : ℕ+) : ℝ × ℝ := (k - 1, 3*k)

/-- The line on which the centers lie -/
def CenterLine (x y : ℝ) : Prop := y = 3*(x + 1) ∧ x ≠ -1

/-- Theorem: If the centers of the circles C_k lie on a fixed line,
    then that line is y = 3(x+1) where x ≠ -1 -/
theorem circle_centers_line :
  (∀ k : ℕ+, ∃ x y : ℝ, CircleCenter k = (x, y) ∧ CenterLine x y) →
  ∀ x y : ℝ, (∃ k : ℕ+, CircleCenter k = (x, y)) → CenterLine x y :=
sorry

end circle_centers_line_l3617_361709


namespace alyssa_car_wash_earnings_l3617_361721

/-- The amount Alyssa earned from washing the family car -/
def car_wash_earnings (weekly_allowance : ℝ) (movie_spending_fraction : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - (weekly_allowance * (1 - movie_spending_fraction))

/-- Theorem: Alyssa earned 8 dollars from washing the family car -/
theorem alyssa_car_wash_earnings :
  car_wash_earnings 8 0.5 12 = 8 := by
  sorry

end alyssa_car_wash_earnings_l3617_361721


namespace perpendicular_vectors_x_value_l3617_361756

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a = (3, 2) →
  b.1 = x →
  b.2 = 4 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = -8/3 := by
sorry

end perpendicular_vectors_x_value_l3617_361756


namespace simplify_expression_l3617_361792

theorem simplify_expression (a : ℝ) : 
  3 * a * (a + 1) - (3 + a) * (3 - a) - (2 * a - 1)^2 = 7 * a - 10 := by
  sorry

end simplify_expression_l3617_361792


namespace line_plane_relations_l3617_361785

/-- The direction vector of line l -/
def m (a b : ℝ) : ℝ × ℝ × ℝ := (1, a + b, a - b)

/-- The normal vector of plane α -/
def n : ℝ × ℝ × ℝ := (1, 2, 3)

/-- Line l is parallel to plane α -/
def is_parallel (a b : ℝ) : Prop :=
  let (x₁, y₁, z₁) := m a b
  let (x₂, y₂, z₂) := n
  x₁ * x₂ + y₁ * y₂ + z₁ * z₂ = 0

/-- Line l is perpendicular to plane α -/
def is_perpendicular (a b : ℝ) : Prop :=
  let (x₁, y₁, z₁) := m a b
  let (x₂, y₂, z₂) := n
  x₁ / x₂ = y₁ / y₂ ∧ x₁ / x₂ = z₁ / z₂

theorem line_plane_relations (a b : ℝ) :
  (is_parallel a b → 5 * a - b + 1 = 0) ∧
  (is_perpendicular a b → a + b - 2 = 0 ∧ a - b - 3 = 0) := by
  sorry

end line_plane_relations_l3617_361785


namespace unit_vectors_equality_iff_sum_magnitude_two_l3617_361772

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem unit_vectors_equality_iff_sum_magnitude_two
  (a b : E) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  a = b ↔ ‖a + b‖ = 2 := by sorry

end unit_vectors_equality_iff_sum_magnitude_two_l3617_361772


namespace two_digit_number_exchange_l3617_361782

theorem two_digit_number_exchange (A : ℕ) : 
  A < 10 →  -- Ensure A is a single digit
  (10 * A + 2) - (20 + A) = 9 →  -- Condition for digit exchange
  A = 3 := by sorry

end two_digit_number_exchange_l3617_361782


namespace complex_multiplication_l3617_361706

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (i + 1) = -1 + i := by sorry

end complex_multiplication_l3617_361706


namespace mixed_coffee_bag_weight_l3617_361761

/-- Proves that the total weight of a mixed coffee bag is 102.8 pounds given specific conditions --/
theorem mixed_coffee_bag_weight 
  (colombian_price : ℝ) 
  (peruvian_price : ℝ) 
  (mixed_price : ℝ) 
  (colombian_weight : ℝ) 
  (h1 : colombian_price = 5.50)
  (h2 : peruvian_price = 4.25)
  (h3 : mixed_price = 4.60)
  (h4 : colombian_weight = 28.8) :
  ∃ (total_weight : ℝ), total_weight = 102.8 ∧ 
  (colombian_price * colombian_weight + peruvian_price * (total_weight - colombian_weight)) / total_weight = mixed_price :=
by
  sorry

#check mixed_coffee_bag_weight

end mixed_coffee_bag_weight_l3617_361761


namespace segment_is_definition_l3617_361777

-- Define the type for geometric statements
inductive GeometricStatement
  | TwoPointsLine
  | SegmentDefinition
  | ComplementaryAngles
  | AlternateInteriorAngles

-- Define a predicate to check if a statement is a definition
def isDefinition : GeometricStatement → Prop
  | GeometricStatement.SegmentDefinition => True
  | _ => False

-- Theorem statement
theorem segment_is_definition :
  (∃! s : GeometricStatement, isDefinition s) →
  isDefinition GeometricStatement.SegmentDefinition :=
by
  sorry

end segment_is_definition_l3617_361777


namespace tangent_line_equation_l3617_361715

-- Define the function f(x) = x^4 - 2x^3
def f (x : ℝ) : ℝ := x^4 - 2*x^3

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 4*x^3 - 6*x^2

-- Theorem: The equation of the tangent line to f(x) at x = 1 is y = -2x + 1
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
by sorry

end tangent_line_equation_l3617_361715


namespace min_shift_value_l3617_361759

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - Real.sqrt 3 * cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (2 * x + π / 3) - Real.sqrt 3 / 2

theorem min_shift_value (k : ℝ) (h : k > 0) :
  (∀ x, f x = g (x - k)) ↔ k ≥ π / 3 :=
sorry

end min_shift_value_l3617_361759


namespace complex_modulus_problem_l3617_361730

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 1) : 
  Complex.abs (2 * z - 3) = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l3617_361730


namespace committee_selection_count_l3617_361749

/-- The number of ways to choose a committee from a club -/
def choose_committee (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

/-- The size of the club -/
def club_size : ℕ := 10

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem: The number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_selection_count : 
  choose_committee club_size committee_size = 252 := by
  sorry

end committee_selection_count_l3617_361749


namespace susan_money_l3617_361747

theorem susan_money (S : ℝ) : 
  S - S/5 - S/4 - 120 = 540 → S = 1200 := by
  sorry

end susan_money_l3617_361747


namespace imaginary_part_of_i_times_one_plus_i_l3617_361790

theorem imaginary_part_of_i_times_one_plus_i (i : ℂ) : 
  i * i = -1 → Complex.im (i * (1 + i)) = 1 := by
sorry

end imaginary_part_of_i_times_one_plus_i_l3617_361790


namespace police_catch_time_police_catch_rogue_l3617_361758

/-- The time it takes for a police spaceship to catch up with a rogue spaceship -/
theorem police_catch_time (rogue_speed : ℝ) (head_start_minutes : ℝ) (police_speed_increase : ℝ) : ℝ :=
  let head_start_hours := head_start_minutes / 60
  let police_speed := rogue_speed * (1 + police_speed_increase)
  let distance_traveled := rogue_speed * head_start_hours
  let relative_speed := police_speed - rogue_speed
  let catch_up_time_hours := distance_traveled / relative_speed
  catch_up_time_hours * 60

/-- The police will catch up with the rogue spaceship in 450 minutes -/
theorem police_catch_rogue : 
  ∀ (rogue_speed : ℝ), rogue_speed > 0 → police_catch_time rogue_speed 54 0.12 = 450 :=
by
  sorry


end police_catch_time_police_catch_rogue_l3617_361758


namespace vectors_are_coplanar_l3617_361724

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  [Fact (finrank ℝ V = 3)]

def are_coplanar (v₁ v₂ v₃ : V) : Prop :=
  ∃ (a b c : ℝ), a • v₁ + b • v₂ + c • v₃ = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

theorem vectors_are_coplanar (a b : V) : are_coplanar a b (2 • a + 4 • b) := by
  sorry

end vectors_are_coplanar_l3617_361724


namespace cricketer_specific_average_l3617_361793

/-- Represents the average score calculation for a cricketer's matches -/
def cricketer_average_score (total_matches : ℕ) (first_set_matches : ℕ) (second_set_matches : ℕ) 
  (first_set_average : ℚ) (second_set_average : ℚ) : ℚ :=
  ((first_set_matches : ℚ) * first_set_average + (second_set_matches : ℚ) * second_set_average) / (total_matches : ℚ)

/-- Theorem stating the average score calculation for a specific cricketer's performance -/
theorem cricketer_specific_average : 
  cricketer_average_score 25 10 15 60 70 = 66 := by
  sorry

end cricketer_specific_average_l3617_361793


namespace owl_cost_in_gold_harry_owl_cost_l3617_361750

/-- Calculates the cost of an owl given the total cost and the cost of other items. -/
theorem owl_cost_in_gold (spellbook_cost : ℕ) (spellbook_count : ℕ) 
  (potion_kit_cost : ℕ) (potion_kit_count : ℕ) (silver_per_gold : ℕ) (total_cost_silver : ℕ) : ℕ :=
  let spellbook_total_cost := spellbook_cost * spellbook_count * silver_per_gold
  let potion_kit_total_cost := potion_kit_cost * potion_kit_count
  let other_items_cost := spellbook_total_cost + potion_kit_total_cost
  let owl_cost_silver := total_cost_silver - other_items_cost
  owl_cost_silver / silver_per_gold

/-- Proves that the owl costs 28 gold given the specific conditions in Harry's purchase. -/
theorem harry_owl_cost : 
  owl_cost_in_gold 5 5 20 3 9 537 = 28 := by
  sorry

end owl_cost_in_gold_harry_owl_cost_l3617_361750


namespace nine_sided_polygon_diagonals_l3617_361734

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n.choose 2 - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals : num_diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l3617_361734


namespace quadratic_roots_isosceles_triangle_l3617_361791

theorem quadratic_roots_isosceles_triangle (b : ℝ) (α β : ℝ) :
  (∀ x, x^2 + b*x + 1 = 0 ↔ x = α ∨ x = β) →
  α > β →
  (α^2 + β^2 = 3*α - 3*β ∧ α^2 + β^2 = α*β) ∨
  (α^2 + β^2 = 3*α - 3*β ∧ 3*α - 3*β = α*β) ∨
  (3*α - 3*β = α*β ∧ α*β = α^2 + β^2) →
  b = Real.sqrt 5 ∨ b = -Real.sqrt 5 ∨ b = Real.sqrt 8 ∨ b = -Real.sqrt 8 := by
sorry

end quadratic_roots_isosceles_triangle_l3617_361791


namespace two_digit_numbers_with_special_properties_l3617_361725

/-- Round a natural number to the nearest ten -/
def roundToNearestTen (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

/-- Check if a natural number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_numbers_with_special_properties (p q : ℕ) :
  isTwoDigit p ∧ isTwoDigit q ∧
  (roundToNearestTen p - roundToNearestTen q = p - q) ∧
  (roundToNearestTen p * roundToNearestTen q = p * q + 184) →
  ((p = 16 ∧ q = 26) ∨ (p = 26 ∧ q = 16)) := by
  sorry

end two_digit_numbers_with_special_properties_l3617_361725


namespace root_sum_reciprocal_l3617_361736

theorem root_sum_reciprocal (p q r : ℝ) : 
  (p^3 - p - 6 = 0) → 
  (q^3 - q - 6 = 0) → 
  (r^3 - r - 6 = 0) → 
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12) := by
sorry

end root_sum_reciprocal_l3617_361736


namespace max_trailing_zeros_product_l3617_361764

/-- Given three natural numbers that sum to 1003, the maximum number of trailing zeros in their product is 7. -/
theorem max_trailing_zeros_product (a b c : ℕ) (h_sum : a + b + c = 1003) :
  (∃ n : ℕ, a * b * c = n * 10^7 ∧ n % 10 ≠ 0) ∧
  ¬(∃ m : ℕ, a * b * c = m * 10^8) :=
by sorry

end max_trailing_zeros_product_l3617_361764


namespace fibonacci_product_theorem_l3617_361735

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sum of squares of divisors -/
def sum_of_squares_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => x * x)

/-- Main theorem -/
theorem fibonacci_product_theorem (N : ℕ) (h_pos : N > 0)
  (h_sum : sum_of_squares_of_divisors N = N * (N + 3)) :
  ∃ i j, N = fib i * fib j :=
sorry

end fibonacci_product_theorem_l3617_361735


namespace problem_solving_probability_l3617_361727

theorem problem_solving_probability 
  (xavier_prob : ℚ) 
  (yvonne_prob : ℚ) 
  (zelda_prob : ℚ) 
  (hx : xavier_prob = 1/6)
  (hy : yvonne_prob = 1/2)
  (hz : zelda_prob = 5/8) :
  xavier_prob * yvonne_prob * (1 - zelda_prob) = 1/32 := by
sorry

end problem_solving_probability_l3617_361727


namespace salary_increase_with_manager_l3617_361753

/-- Proves that adding a manager's salary increases the average salary by 150 rupees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 15 → 
  avg_salary = 1800 → 
  manager_salary = 4200 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 150 := by
  sorry

end salary_increase_with_manager_l3617_361753


namespace rachel_age_problem_l3617_361740

/-- Rachel's age problem -/
theorem rachel_age_problem (rachel_age : ℕ) (grandfather_age : ℕ) (mother_age : ℕ) (father_age : ℕ) : 
  rachel_age = 12 →
  grandfather_age = 7 * rachel_age →
  mother_age = grandfather_age / 2 →
  father_age = mother_age + 5 →
  father_age + (25 - rachel_age) = 60 :=
by sorry

end rachel_age_problem_l3617_361740


namespace jim_has_220_buicks_l3617_361732

/-- Represents the number of model cars of each brand Jim has. -/
structure ModelCars where
  ford : ℕ
  buick : ℕ
  chevy : ℕ

/-- The conditions of Jim's model car collection. -/
def jim_collection (cars : ModelCars) : Prop :=
  cars.ford + cars.buick + cars.chevy = 301 ∧
  cars.buick = 4 * cars.ford ∧
  cars.ford = 2 * cars.chevy + 3

/-- Theorem stating that Jim has 220 Buicks. -/
theorem jim_has_220_buicks :
  ∃ (cars : ModelCars), jim_collection cars ∧ cars.buick = 220 := by
  sorry

end jim_has_220_buicks_l3617_361732


namespace book_reading_fraction_l3617_361726

theorem book_reading_fraction (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 300 →
  pages_read = (total_pages - pages_read) + 100 →
  pages_read / total_pages = 2 / 3 := by
  sorry

end book_reading_fraction_l3617_361726


namespace concave_probability_is_one_third_l3617_361711

/-- A digit is a natural number between 4 and 8 inclusive -/
def Digit : Type := { n : ℕ // 4 ≤ n ∧ n ≤ 8 }

/-- A three-digit number is a tuple of three digits -/
def ThreeDigitNumber : Type := Digit × Digit × Digit

/-- A concave number is a three-digit number where the first and third digits are greater than the second -/
def is_concave (n : ThreeDigitNumber) : Prop :=
  let (a, b, c) := n
  a.val > b.val ∧ c.val > b.val

/-- The set of all possible three-digit numbers with distinct digits from {4,5,6,7,8} -/
def all_numbers : Finset ThreeDigitNumber :=
  sorry

/-- The set of all concave numbers from all_numbers -/
def concave_numbers : Finset ThreeDigitNumber :=
  sorry

/-- The probability of a randomly chosen three-digit number being concave -/
def concave_probability : ℚ :=
  (Finset.card concave_numbers : ℚ) / (Finset.card all_numbers : ℚ)

theorem concave_probability_is_one_third :
  concave_probability = 1 / 3 :=
sorry

end concave_probability_is_one_third_l3617_361711


namespace paper_I_max_mark_l3617_361744

/-- Represents a test with a maximum mark and a passing percentage -/
structure Test where
  maxMark : ℕ
  passingPercentage : ℚ

/-- Calculates the passing mark for a given test -/
def passingMark (test : Test) : ℚ :=
  test.passingPercentage * test.maxMark

theorem paper_I_max_mark :
  ∃ (test : Test),
    test.passingPercentage = 42 / 100 ∧
    passingMark test = 42 + 22 ∧
    test.maxMark = 152 := by
  sorry

end paper_I_max_mark_l3617_361744


namespace purple_ribbons_l3617_361712

theorem purple_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 6 →
  black = 40 →
  yellow + purple + orange + black = total →
  purple = 53 := by
sorry

end purple_ribbons_l3617_361712


namespace max_abs_z_value_l3617_361705

/-- Given complex numbers a, b, c, z satisfying the conditions, 
    the maximum value of |z| is 1 + √2 -/
theorem max_abs_z_value (a b c z : ℂ) (r : ℝ) 
  (hr : r > 0)
  (ha : Complex.abs a = r)
  (hb : Complex.abs b = 2*r)
  (hc : Complex.abs c = r)
  (heq : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ 1 + Real.sqrt 2 :=
sorry

end max_abs_z_value_l3617_361705


namespace number_triples_satisfying_equation_l3617_361774

theorem number_triples_satisfying_equation :
  ∀ (a b c : ℕ), a^(b+20) * (c-1) = c^(b+21) - 1 ↔ 
    ((a = 1 ∧ c = 0) ∨ c = 1) :=
by sorry

end number_triples_satisfying_equation_l3617_361774


namespace rectangle_length_l3617_361746

-- Define the radius of the circle
def R : ℝ := 2.5

-- Define pi as an approximation
def π : ℝ := 3.14

-- Define the perimeter of the rectangle
def perimeter : ℝ := 20.7

-- Theorem stating that the length of the rectangle is 7.85 cm
theorem rectangle_length : (π * R) = 7.85 := by
  sorry

end rectangle_length_l3617_361746


namespace distribution_methods_eq_72_l3617_361722

/-- Number of teachers -/
def num_teachers : ℕ := 3

/-- Number of students -/
def num_students : ℕ := 3

/-- Total number of tickets -/
def total_tickets : ℕ := 6

/-- Function to calculate the number of distribution methods -/
def distribution_methods : ℕ := sorry

/-- Theorem stating that the number of distribution methods is 72 -/
theorem distribution_methods_eq_72 : distribution_methods = 72 := by sorry

end distribution_methods_eq_72_l3617_361722


namespace system_solution_l3617_361741

theorem system_solution (x y z : ℝ) (eq1 : x + y + z = 0) (eq2 : 4 * x + 2 * y + z = 0) :
  y = -3 * x ∧ z = 2 * x := by
sorry

end system_solution_l3617_361741


namespace combined_mix_dried_fruit_percentage_l3617_361760

/-- Represents a trail mix composition -/
structure TrailMix where
  nuts : ℚ
  dried_fruit : ℚ
  chocolate_chips : ℚ
  composition_sum : nuts + dried_fruit + chocolate_chips = 1

/-- The combined trail mix from two equal portions -/
def combined_mix (mix1 mix2 : TrailMix) : TrailMix :=
  { nuts := (mix1.nuts + mix2.nuts) / 2,
    dried_fruit := (mix1.dried_fruit + mix2.dried_fruit) / 2,
    chocolate_chips := (mix1.chocolate_chips + mix2.chocolate_chips) / 2,
    composition_sum := by sorry }

theorem combined_mix_dried_fruit_percentage
  (sue_mix : TrailMix)
  (jane_mix : TrailMix)
  (h_sue_nuts : sue_mix.nuts = 3/10)
  (h_sue_dried_fruit : sue_mix.dried_fruit = 7/10)
  (h_jane_nuts : jane_mix.nuts = 6/10)
  (h_jane_chocolate : jane_mix.chocolate_chips = 4/10)
  (h_combined_nuts : (combined_mix sue_mix jane_mix).nuts = 45/100) :
  (combined_mix sue_mix jane_mix).dried_fruit = 35/100 := by
sorry

end combined_mix_dried_fruit_percentage_l3617_361760


namespace volume_ratio_l3617_361728

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) :
  C / (A + B) = 23 / 12 := by
sorry

end volume_ratio_l3617_361728


namespace work_completion_time_l3617_361786

theorem work_completion_time (a_time b_time : ℝ) (a_share : ℝ) : 
  a_time = 10 →
  a_share = 3 / 5 →
  a_share = (1 / a_time) / ((1 / a_time) + (1 / b_time)) →
  b_time = 15 := by
  sorry

end work_completion_time_l3617_361786


namespace trees_in_garden_l3617_361766

/-- Given a yard of length 600 meters with trees planted at equal distances,
    including one at each end, and a distance of 24 meters between consecutive trees,
    the total number of trees planted is 26. -/
theorem trees_in_garden (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 600) (h2 : tree_distance = 24) :
  yard_length / tree_distance + 1 = 26 := by
  sorry

end trees_in_garden_l3617_361766


namespace work_days_solution_l3617_361737

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person b -/
def days_b : ℕ := 9

/-- The number of days worked by person c -/
def days_c : ℕ := 4

/-- The daily wage of person c -/
def wage_c : ℕ := 100

/-- The total earnings of all three workers -/
def total_earnings : ℕ := 1480

/-- The ratio of daily wages for a, b, and c respectively -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

theorem work_days_solution :
  ∃ (wage_a wage_b : ℕ),
    wage_a = wage_ratio 0 * (wage_c / wage_ratio 2) ∧
    wage_b = wage_ratio 1 * (wage_c / wage_ratio 2) ∧
    wage_a * days_a + wage_b * days_b + wage_c * days_c = total_earnings :=
by sorry


end work_days_solution_l3617_361737


namespace motorcycle_speeds_correct_l3617_361776

/-- Two motorcyclists travel towards each other with uniform speed. -/
structure MotorcycleJourney where
  /-- Total distance between starting points A and B in km -/
  total_distance : ℝ
  /-- Distance traveled by the first motorcyclist when the second has traveled 200 km -/
  first_partial_distance : ℝ
  /-- Time difference in hours between arrivals -/
  time_difference : ℝ
  /-- Speed of the first motorcyclist in km/h -/
  speed_first : ℝ
  /-- Speed of the second motorcyclist in km/h -/
  speed_second : ℝ

/-- The speeds of the motorcyclists satisfy the given conditions -/
def satisfies_conditions (j : MotorcycleJourney) : Prop :=
  j.total_distance = 600 ∧
  j.first_partial_distance = 250 ∧
  j.time_difference = 3 ∧
  j.first_partial_distance / j.speed_first = 200 / j.speed_second ∧
  j.total_distance / j.speed_first + j.time_difference = j.total_distance / j.speed_second

/-- The theorem stating that the given speeds satisfy the conditions -/
theorem motorcycle_speeds_correct (j : MotorcycleJourney) :
  j.speed_first = 50 ∧ j.speed_second = 40 → satisfies_conditions j :=
by sorry

end motorcycle_speeds_correct_l3617_361776


namespace novel_writing_rate_l3617_361702

/-- Represents the writing rate of an author -/
def writing_rate (total_words : ℕ) (writing_hours : ℕ) : ℚ :=
  total_words / writing_hours

/-- Proves that the writing rate for a 60,000-word novel written in 100 hours is 600 words per hour -/
theorem novel_writing_rate :
  writing_rate 60000 100 = 600 := by
  sorry

end novel_writing_rate_l3617_361702


namespace number_reciprocal_problem_l3617_361701

theorem number_reciprocal_problem (x : ℝ) (h : 8 * x - 6 = 10) :
  50 * (1 / x) + 150 = 175 := by
  sorry

end number_reciprocal_problem_l3617_361701


namespace ellipse_proof_l3617_361767

-- Define the given ellipse
def given_ellipse (x y : ℝ) : Prop := 9 * x^2 + 4 * y^2 = 36

-- Define the equation of the ellipse we want to prove
def target_ellipse (x y : ℝ) : Prop := y^2/16 + x^2/11 = 1

-- Theorem statement
theorem ellipse_proof :
  -- The target ellipse passes through (0, 4)
  target_ellipse 0 4 ∧
  -- The target ellipse has the same foci as the given ellipse
  ∃ (c : ℝ), c^2 = 5 ∧
    ∀ (x y : ℝ), given_ellipse x y ↔ 
      ∃ (a b : ℝ), a^2 = 9 ∧ b^2 = 4 ∧ c^2 = a^2 - b^2 ∧ y^2/a^2 + x^2/b^2 = 1 :=
by sorry

end ellipse_proof_l3617_361767


namespace andrew_payment_l3617_361752

/-- The total amount Andrew paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: Andrew paid 1055 to the shopkeeper -/
theorem andrew_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end andrew_payment_l3617_361752


namespace security_system_connections_l3617_361763

/-- 
Given a security system with 25 switches where each switch is connected to exactly 4 other switches,
the total number of connections is 50.
-/
theorem security_system_connections (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n * k) / 2 = 50 := by
  sorry

end security_system_connections_l3617_361763


namespace root_preservation_l3617_361748

/-- Given a polynomial P(x) = x^3 + ax^2 + bx + c with three distinct real roots,
    the polynomial Q(x) = x^3 + ax^2 + (1/4)(a^2 + b)x + (1/8)(ab - c) also has three distinct real roots. -/
theorem root_preservation (a b c : ℝ) 
  (h : ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :
  ∃ (y₁ y₂ y₃ : ℝ), y₁ ≠ y₂ ∧ y₂ ≠ y₃ ∧ y₁ ≠ y₃ ∧
    (∀ x, x^3 + a*x^2 + (1/4)*(a^2 + b)*x + (1/8)*(a*b - c) = 0 ↔ x = y₁ ∨ x = y₂ ∨ x = y₃) :=
by sorry

end root_preservation_l3617_361748


namespace five_digit_divisible_by_nine_l3617_361798

theorem five_digit_divisible_by_nine :
  ∃! d : ℕ, d < 10 ∧ (34700 + 10 * d + 9) % 9 = 0 := by
  sorry

end five_digit_divisible_by_nine_l3617_361798


namespace polynomial_coefficient_properties_l3617_361796

theorem polynomial_coefficient_properties (a : Fin 6 → ℝ) :
  (∀ x : ℝ, x^5 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5) →
  (a 3 = -10 ∧ a 1 + a 3 + a 5 = -16) := by
  sorry

end polynomial_coefficient_properties_l3617_361796


namespace parabola_transformation_sum_of_zeros_l3617_361784

/-- Represents a parabola and its transformations -/
structure Parabola where
  a : ℝ  -- coefficient of x^2
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex

/-- Apply transformations to the parabola -/
def transform (p : Parabola) : Parabola :=
  { a := -p.a,  -- 180-degree rotation
    h := p.h + 4,  -- 4 units right shift
    k := p.k + 4 }  -- 4 units up shift

/-- Calculate the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := 2 * p.h

theorem parabola_transformation_sum_of_zeros :
  let original := Parabola.mk 1 2 3
  let transformed := transform original
  sumOfZeros transformed = 12 := by sorry

end parabola_transformation_sum_of_zeros_l3617_361784


namespace abs_even_and_decreasing_l3617_361771

def f (x : ℝ) := abs x

theorem abs_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y ≤ f x) :=
by sorry

end abs_even_and_decreasing_l3617_361771


namespace similar_triangles_height_l3617_361768

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_small = 5 →
    h_large = 15 := by
  sorry

end similar_triangles_height_l3617_361768


namespace some_ounce_glass_size_l3617_361781

/-- Proves that the size of the some-ounce glasses is 5 ounces given the problem conditions. -/
theorem some_ounce_glass_size (total_water : ℕ) (S : ℕ) 
  (h1 : total_water = 122)
  (h2 : 6 * S + 4 * 8 + 15 * 4 = total_water) : S = 5 := by
  sorry

#check some_ounce_glass_size

end some_ounce_glass_size_l3617_361781


namespace sequence_existence_l3617_361704

theorem sequence_existence : ∃ (a b : ℕ → ℕ), 
  (∀ n : ℕ, n ≥ 1 → (
    (0 < a n ∧ a n < a (n + 1)) ∧
    (a n < b n ∧ b n < a n ^ 2) ∧
    ((b n - 1) % (a n - 1) = 0) ∧
    ((b n ^ 2 - 1) % (a n ^ 2 - 1) = 0)
  )) := by
  sorry

end sequence_existence_l3617_361704


namespace right_angled_triangle_l3617_361751

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)

/-- The theorem stating that if certain conditions are met, the triangle is right-angled. -/
theorem right_angled_triangle (t : Triangle) 
  (h1 : (Real.sqrt 3 * t.c) / (t.a * Real.cos t.B) = Real.tan t.A + Real.tan t.B)
  (h2 : t.b - t.c = (Real.sqrt 3 * t.a) / 3) : 
  t.B = π / 2 := by
  sorry

end right_angled_triangle_l3617_361751


namespace arun_weight_average_l3617_361723

theorem arun_weight_average :
  let min_weight := 61
  let max_weight := 64
  let average := (min_weight + max_weight) / 2
  (∀ w, min_weight < w ∧ w ≤ max_weight → 
    w > 60 ∧ w < 70 ∧ w > 61 ∧ w < 72 ∧ w ≤ 64) →
  average = 62.5 := by
sorry

end arun_weight_average_l3617_361723


namespace quadratic_root_geometric_sequence_l3617_361731

theorem quadratic_root_geometric_sequence (a b c : ℝ) : 
  a ≥ b ∧ b ≥ c ∧ c ≥ 0 →  -- Condition: a ≥ b ≥ c ≥ 0
  (∃ r : ℝ, b = a * r ∧ c = a * r^2) →  -- Condition: a, b, c form a geometric sequence
  (∃! x : ℝ, a * x^2 + b * x + c = 0) →  -- Condition: quadratic has exactly one root
  (∀ x : ℝ, a * x^2 + b * x + c = 0 → x = -1/8) :=  -- Conclusion: the root is -1/8
by sorry

end quadratic_root_geometric_sequence_l3617_361731


namespace similar_triangle_sum_l3617_361738

/-- Given a triangle with sides in ratio 3:5:7 and a similar triangle with longest side 21,
    the sum of the other two sides of the similar triangle is 24. -/
theorem similar_triangle_sum (a b c : ℝ) (x y z : ℝ) :
  a / b = 3 / 5 →
  b / c = 5 / 7 →
  a / c = 3 / 7 →
  x / y = a / b →
  y / z = b / c →
  x / z = a / c →
  z = 21 →
  x + y = 24 := by
sorry

end similar_triangle_sum_l3617_361738


namespace distribute_six_balls_three_boxes_l3617_361718

/-- Number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute 6 3 = 92 := by sorry

end distribute_six_balls_three_boxes_l3617_361718


namespace uncertain_mushrooms_l3617_361770

theorem uncertain_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) (uncertain : ℕ) : 
  total = 32 → 
  safe = 9 → 
  poisonous = 2 * safe → 
  total = safe + poisonous + uncertain → 
  uncertain = 5 := by
sorry

end uncertain_mushrooms_l3617_361770


namespace students_in_both_clubs_l3617_361757

def total_students : ℕ := 250
def drama_club : ℕ := 80
def science_club : ℕ := 120
def either_or_both : ℕ := 180

theorem students_in_both_clubs :
  ∃ (both : ℕ), both = drama_club + science_club - either_or_both ∧ both = 20 := by
  sorry

end students_in_both_clubs_l3617_361757


namespace puppies_percentage_proof_l3617_361719

/-- The percentage of students who have puppies in Professor Plum's biology class -/
def percentage_with_puppies : ℝ := 80

theorem puppies_percentage_proof (total_students : ℕ) (both_puppies_parrots : ℕ) 
  (h1 : total_students = 40)
  (h2 : both_puppies_parrots = 8)
  (h3 : (25 : ℝ) / 100 * (percentage_with_puppies / 100 * total_students) = both_puppies_parrots) :
  percentage_with_puppies = 80 := by
  sorry

#check puppies_percentage_proof

end puppies_percentage_proof_l3617_361719


namespace new_girl_weight_l3617_361716

/-- Given a group of 25 girls, if replacing a 55 kg girl with a new girl increases
    the average weight by 1 kg, then the new girl weighs 80 kg. -/
theorem new_girl_weight (W : ℝ) (x : ℝ) : 
  (W / 25 + 1 = (W - 55 + x) / 25) → x = 80 := by
  sorry

end new_girl_weight_l3617_361716


namespace odd_function_extension_l3617_361779

/-- An odd function on the real line. -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem odd_function_extension (f : ℝ → ℝ) (h : OddFunction f) 
    (h_neg : ∀ x < 0, f x = x * Real.exp (-x)) :
    ∀ x > 0, f x = x * Real.exp x := by
  sorry

end odd_function_extension_l3617_361779


namespace sequence_explicit_formula_l3617_361778

theorem sequence_explicit_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 2 * a n + 1) →
  ∀ n : ℕ, n ≥ 1 → a n = (-2) ^ (n - 1) := by
  sorry

end sequence_explicit_formula_l3617_361778


namespace arithmetic_sequence_solution_l3617_361733

/-- An arithmetic sequence with first term a, common difference d, and index n -/
def arithmeticSequence (a d : ℚ) (n : ℕ) : ℚ := a + d * n

theorem arithmetic_sequence_solution :
  ∃ (x : ℚ),
    (arithmeticSequence (3/4) d 0 = 3/4) ∧
    (arithmeticSequence (3/4) d 1 = x + 1) ∧
    (arithmeticSequence (3/4) d 2 = 5*x) →
    x = 5/12 := by
  sorry

end arithmetic_sequence_solution_l3617_361733


namespace smallest_prime_dividing_sum_l3617_361788

theorem smallest_prime_dividing_sum : Nat.minFac (7^7 + 3^14) = 2 := by sorry

end smallest_prime_dividing_sum_l3617_361788


namespace decimal_is_fraction_l3617_361743

def is_fraction (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem decimal_is_fraction :
  let x : ℝ := 0.666
  is_fraction x :=
sorry

end decimal_is_fraction_l3617_361743


namespace arithmetic_simplification_l3617_361754

theorem arithmetic_simplification :
  -11 - (-8) + (-13) + 12 = -4 := by
  sorry

end arithmetic_simplification_l3617_361754


namespace digit_1234_is_4_l3617_361710

/-- The decimal number constructed by concatenating integers from 1 to 499 -/
def x : ℚ :=
  sorry

/-- The nth digit of a rational number -/
def nthDigit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_1234_is_4 : nthDigit x 1234 = 4 := by
  sorry

end digit_1234_is_4_l3617_361710


namespace inequality_implies_upper_bound_l3617_361780

theorem inequality_implies_upper_bound (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ a) → a ≤ 3 := by
sorry

end inequality_implies_upper_bound_l3617_361780


namespace erased_number_l3617_361717

theorem erased_number (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 45 / 4 →
  x = 6 := by
  sorry

end erased_number_l3617_361717


namespace total_seeds_calculation_l3617_361755

/-- The number of rows of potatoes planted -/
def rows : ℕ := 6

/-- The number of seeds in each row -/
def seeds_per_row : ℕ := 9

/-- The total number of potato seeds planted -/
def total_seeds : ℕ := rows * seeds_per_row

theorem total_seeds_calculation : total_seeds = 54 := by
  sorry

end total_seeds_calculation_l3617_361755


namespace tan_sum_equals_one_l3617_361742

theorem tan_sum_equals_one (α β : ℝ) 
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 := by
  sorry

end tan_sum_equals_one_l3617_361742


namespace line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l3617_361713

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_to_plane : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained_in_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_to_line : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_implies_perpendicular_to_contained_line 
  (l m : Line) (α : Plane) :
  perpendicular_to_plane l α → contained_in_plane m α → perpendicular_to_line l m :=
by sorry

end line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l3617_361713


namespace triangle_area_l3617_361795

theorem triangle_area (a c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : c = 2 * Real.sqrt 3) (h3 : A = π / 6) :
  ∃ (area : ℝ), (area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3) ∧
  ∃ (B C : ℝ), 0 ≤ B ∧ B < 2 * π ∧ 0 ≤ C ∧ C < 2 * π ∧
  A + B + C = π ∧
  area = (1 / 2) * a * c * Real.sin B :=
by sorry

end triangle_area_l3617_361795


namespace unique_condition_result_l3617_361769

theorem unique_condition_result : ∃ (a b c : ℕ),
  ({a, b, c} : Set ℕ) = {0, 1, 2} ∧
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨
   ((a ≠ 2) ∧ (b = 2) ∧ (c = 0)) ∨
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0))) →
  100 * a + 10 * b + c = 201 :=
by sorry

end unique_condition_result_l3617_361769


namespace min_value_mn_l3617_361799

def f (a x : ℝ) : ℝ := |x - a|

theorem min_value_mn (a m n : ℝ) : 
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  m > 0 →
  n > 0 →
  1/m + 1/(2*n) = a →
  ∀ k, m * n ≤ k → 2 ≤ k :=
by sorry

end min_value_mn_l3617_361799


namespace supermarket_max_profit_l3617_361703

/-- Represents the daily profit function for the supermarket -/
def daily_profit (x : ℝ) : ℝ := -10 * x^2 + 1100 * x - 28000

/-- The maximum daily profit achievable by the supermarket -/
def max_profit : ℝ := 2250

theorem supermarket_max_profit :
  ∃ (x : ℝ), daily_profit x = max_profit ∧
  ∀ (y : ℝ), daily_profit y ≤ max_profit := by
  sorry

#check supermarket_max_profit

end supermarket_max_profit_l3617_361703


namespace reciprocal_equation_l3617_361765

theorem reciprocal_equation (x : ℝ) : 1 - 1 / (1 - x) = 1 / (1 - x) → x = -1 := by
  sorry

end reciprocal_equation_l3617_361765


namespace number_times_five_equals_hundred_l3617_361707

theorem number_times_five_equals_hundred :
  ∃ x : ℝ, 5 * x = 100 ∧ x = 20 := by
  sorry

end number_times_five_equals_hundred_l3617_361707


namespace bottle_caps_difference_l3617_361739

/-- Represents the number of bottle caps in various states of Danny's collection --/
structure BottleCaps where
  thrown_away : ℕ
  found : ℕ
  final_count : ℕ

/-- Theorem stating the difference between found and thrown away bottle caps --/
theorem bottle_caps_difference (caps : BottleCaps)
  (h1 : caps.thrown_away = 6)
  (h2 : caps.found = 50)
  (h3 : caps.final_count = 60)
  : caps.found - caps.thrown_away = 44 := by
  sorry

#check bottle_caps_difference

end bottle_caps_difference_l3617_361739


namespace combination_equality_l3617_361787

theorem combination_equality (n : ℕ) : 
  Nat.choose 5 2 = Nat.choose 5 n → n = 2 ∨ n = 3 := by
  sorry

end combination_equality_l3617_361787


namespace units_digit_of_product_division_l3617_361789

theorem units_digit_of_product_division : 
  (15 * 16 * 17 * 18 * 19 * 20) / 500 % 10 = 8 := by sorry

end units_digit_of_product_division_l3617_361789


namespace triangle_properties_l3617_361720

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the main results -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = 2 * t.B) : 
  (t.b = 2 ∧ t.c = 1 → t.a = Real.sqrt 6) ∧
  (t.b + t.c = Real.sqrt 3 * t.a → t.B = π / 6) := by
  sorry

end triangle_properties_l3617_361720
