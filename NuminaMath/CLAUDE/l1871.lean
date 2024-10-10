import Mathlib

namespace usual_time_calculation_l1871_187149

/-- Given a man who walks at P% of his usual speed and takes T minutes more than usual,
    his usual time U (in minutes) to cover the distance is (P * T) / (100 - P). -/
theorem usual_time_calculation (P T : ℝ) (h1 : 0 < P) (h2 : P < 100) (h3 : 0 < T) :
  ∃ U : ℝ, U > 0 ∧ U = (P * T) / (100 - P) :=
sorry

end usual_time_calculation_l1871_187149


namespace hattie_jumps_l1871_187120

theorem hattie_jumps (H : ℚ) 
  (total_jumps : H + (3/4 * H) + (2/3 * H) + (2/3 * H + 50) = 605) : 
  H = 180 := by
sorry

end hattie_jumps_l1871_187120


namespace a_range_characterization_l1871_187174

-- Define the function p
def p (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- Define the monotonicity condition for p
def p_monotone (a : ℝ) : Prop :=
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < -2 → p a x₁ < p a x₂) ∧
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → p a x₁ < p a x₂)

-- Define the integral function
def integral (x : ℝ) : ℝ := x^2 - 2*x

-- Define the condition for q
def q (a : ℝ) : Prop := ∀ x, integral x > a

-- Define the range of a
def a_range (a : ℝ) : Prop := a < -2 ∨ (-1 ≤ a ∧ a ≤ 2)

-- Theorem statement
theorem a_range_characterization :
  ∀ a : ℝ, (p_monotone a ∧ ¬q a) ∨ (¬p_monotone a ∧ q a) ↔ a_range a :=
sorry

end a_range_characterization_l1871_187174


namespace absolute_difference_sum_product_l1871_187139

theorem absolute_difference_sum_product (x y : ℝ) (hx : x = 12) (hy : y = 18) :
  |x - y| * (x + y) = 180 := by sorry

end absolute_difference_sum_product_l1871_187139


namespace gigi_remaining_pieces_l1871_187153

/-- The number of remaining mushroom pieces after cutting and using some -/
def remaining_pieces (total_mushrooms : ℕ) (pieces_per_mushroom : ℕ) 
  (used_by_kenny : ℕ) (used_by_karla : ℕ) : ℕ :=
  total_mushrooms * pieces_per_mushroom - (used_by_kenny + used_by_karla)

/-- Theorem stating the number of remaining mushroom pieces in GiGi's scenario -/
theorem gigi_remaining_pieces : 
  remaining_pieces 22 4 38 42 = 8 := by sorry

end gigi_remaining_pieces_l1871_187153


namespace min_value_expression_l1871_187193

theorem min_value_expression (a b : ℝ) (h : a^2 * b^2 + 2*a*b + 2*a + 1 = 0) :
  ∃ (x : ℝ), x = a*b*(a*b+2) + (b+1)^2 + 2*a ∧ 
  (∀ (y : ℝ), y = a*b*(a*b+2) + (b+1)^2 + 2*a → x ≤ y) ∧
  x = -3/4 :=
sorry

end min_value_expression_l1871_187193


namespace gcd_7654321_6543210_l1871_187116

theorem gcd_7654321_6543210 : Nat.gcd 7654321 6543210 = 1 := by
  sorry

end gcd_7654321_6543210_l1871_187116


namespace fifth_number_in_first_set_l1871_187124

theorem fifth_number_in_first_set (x : ℝ) (fifth_number : ℝ) : 
  ((28 + x + 70 + 88 + fifth_number) / 5 = 67) →
  ((50 + 62 + 97 + 124 + x) / 5 = 75.6) →
  fifth_number = 104 := by
  sorry

end fifth_number_in_first_set_l1871_187124


namespace rectangle_area_l1871_187104

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let p := 2 * (l + b)
  p = 112 → l * b = 588 := by sorry

end rectangle_area_l1871_187104


namespace multiple_of_nine_implies_multiple_of_three_l1871_187150

theorem multiple_of_nine_implies_multiple_of_three (n : ℤ) :
  (∀ k : ℤ, 9 ∣ k → 3 ∣ k) →
  9 ∣ n →
  3 ∣ n := by
  sorry

end multiple_of_nine_implies_multiple_of_three_l1871_187150


namespace football_team_addition_l1871_187121

theorem football_team_addition : 36 + 14 = 50 := by
  sorry

end football_team_addition_l1871_187121


namespace complex_multiplication_l1871_187173

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end complex_multiplication_l1871_187173


namespace box_third_side_length_l1871_187189

/-- Proves that the third side of a rectangular box is 6.75 cm given specific conditions -/
theorem box_third_side_length (num_cubes : ℕ) (cube_volume : ℝ) (side1 side2 : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  side1 = 8 →
  side2 = 12 →
  (num_cubes : ℝ) * cube_volume = side1 * side2 * 6.75 :=
by sorry

end box_third_side_length_l1871_187189


namespace slope_intercept_product_specific_line_l1871_187177

/-- A line in a cartesian plane. -/
structure Line where
  /-- The slope of the line. -/
  slope : ℝ
  /-- The y-intercept of the line. -/
  y_intercept : ℝ

/-- The product of the slope and y-intercept of a line. -/
def slope_intercept_product (l : Line) : ℝ := l.slope * l.y_intercept

/-- Theorem: For a line with y-intercept -3 and slope 3, the product of its slope and y-intercept is -9. -/
theorem slope_intercept_product_specific_line :
  ∃ (l : Line), l.y_intercept = -3 ∧ l.slope = 3 ∧ slope_intercept_product l = -9 := by
  sorry

end slope_intercept_product_specific_line_l1871_187177


namespace logarithm_sum_simplification_l1871_187156

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 20 + 1) +
  1 / (Real.log 4 / Real.log 15 + 1) +
  1 / (Real.log 7 / Real.log 12 + 1) = 4 := by
  sorry

end logarithm_sum_simplification_l1871_187156


namespace sum_of_coefficients_l1871_187182

/-- A function f(x) = ax^2 + bx + 1 that is even and has domain [2a, 1-a] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The domain of f is [2a, 1-a] -/
def domain (a : ℝ) : Set ℝ := Set.Icc (2 * a) (1 - a)

/-- f is an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem sum_of_coefficients (a b : ℝ) :
  (∃ x, x ∈ domain a) →
  is_even (f a b) →
  a + b = -1 :=
sorry

end sum_of_coefficients_l1871_187182


namespace external_tangent_lines_of_circles_l1871_187199

-- Define the circles
def circle_A (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 9
def circle_B (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the external common tangent lines
def external_tangent_lines (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x - 3) ∨ y = -(Real.sqrt 3 / 3) * (x - 3)

-- Theorem statement
theorem external_tangent_lines_of_circles :
  ∀ x y : ℝ, (circle_A x y ∨ circle_B x y) → external_tangent_lines x y :=
by
  sorry

end external_tangent_lines_of_circles_l1871_187199


namespace restaurant_bill_rounding_l1871_187100

theorem restaurant_bill_rounding (people : ℕ) (bill : ℚ) : 
  people = 9 → 
  bill = 314.16 → 
  ∃ (rounded_total : ℚ), 
    rounded_total = (people : ℚ) * (⌈(bill / people) * 100⌉ / 100) ∧ 
    rounded_total = 314.19 := by
sorry

end restaurant_bill_rounding_l1871_187100


namespace negative_product_cube_squared_l1871_187181

theorem negative_product_cube_squared (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end negative_product_cube_squared_l1871_187181


namespace fraction_equality_l1871_187138

theorem fraction_equality : (25 + 15) / (5 - 3) = 20 := by
  sorry

end fraction_equality_l1871_187138


namespace intersection_of_A_and_B_l1871_187146

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 4} := by
  sorry

end intersection_of_A_and_B_l1871_187146


namespace fourth_fifth_supplier_cars_l1871_187145

/-- The number of cars each of the fourth and fifth suppliers receive -/
def cars_per_last_supplier (total_cars : ℕ) (first_supplier : ℕ) (additional_second : ℕ) : ℕ :=
  let second_supplier := first_supplier + additional_second
  let third_supplier := first_supplier + second_supplier
  let remaining_cars := total_cars - (first_supplier + second_supplier + third_supplier)
  remaining_cars / 2

/-- Proof that given the conditions, the fourth and fifth suppliers each receive 325,000 cars -/
theorem fourth_fifth_supplier_cars :
  cars_per_last_supplier 5650000 1000000 500000 = 325000 := by
  sorry

end fourth_fifth_supplier_cars_l1871_187145


namespace ab_inequality_l1871_187178

theorem ab_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := by
  sorry

end ab_inequality_l1871_187178


namespace intersection_M_N_l1871_187105

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x | -1 < x ∧ x < 4}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end intersection_M_N_l1871_187105


namespace tetrahedron_surface_area_l1871_187107

/-- The surface area of a regular tetrahedron inscribed in a sphere, 
    which is itself inscribed in a cube with a surface area of 54 square meters. -/
theorem tetrahedron_surface_area : ℝ := by
  -- Define the surface area of the cube
  let cube_surface_area : ℝ := 54

  -- Define the relationship between the cube and the inscribed sphere
  let sphere_inscribed_in_cube : Prop := sorry

  -- Define the relationship between the sphere and the inscribed tetrahedron
  let tetrahedron_inscribed_in_sphere : Prop := sorry

  -- State that the surface area of the inscribed regular tetrahedron is 12√3
  have h : ∃ (area : ℝ), area = 12 * Real.sqrt 3 := sorry

  -- The actual proof would go here
  sorry

end tetrahedron_surface_area_l1871_187107


namespace triangle_existence_l1871_187140

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Calculates the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ :=
  sorry

/-- Checks if an angle is obtuse -/
def isObtuse (θ : ℝ) : Prop :=
  θ > Real.pi / 2

/-- Constructs a triangle given A₀, A₁, and A₂ -/
noncomputable def constructTriangle (A₀ A₁ A₂ : Point) : Option Triangle :=
  sorry

theorem triangle_existence (A₀ A₁ A₂ : Point) :
  ¬collinear A₀ A₁ A₂ →
  isObtuse (angle A₀ A₁ A₂) →
  ∃! (t : Triangle),
    (constructTriangle A₀ A₁ A₂ = some t) ∧
    (A₀.x = (t.B.x + t.C.x) / 2 ∧ A₀.y = (t.B.y + t.C.y) / 2) ∧
    collinear A₁ t.B t.C ∧
    (let midAlt := Point.mk ((t.A.x + A₀.x) / 2) ((t.A.y + A₀.y) / 2);
     A₂ = midAlt) :=
by
  sorry

end triangle_existence_l1871_187140


namespace maurice_age_l1871_187114

theorem maurice_age (ron_age : ℕ) (maurice_age : ℕ) : 
  ron_age = 43 → 
  ron_age + 5 = 4 * (maurice_age + 5) → 
  maurice_age = 7 := by
sorry

end maurice_age_l1871_187114


namespace smallest_prime_factor_of_2379_l1871_187144

theorem smallest_prime_factor_of_2379 : Nat.minFac 2379 = 3 := by
  sorry

end smallest_prime_factor_of_2379_l1871_187144


namespace same_terminal_side_as_405_degrees_l1871_187158

theorem same_terminal_side_as_405_degrees :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 + 45) ↔ (∃ n : ℤ, θ = 405 + n * 360) :=
by sorry

end same_terminal_side_as_405_degrees_l1871_187158


namespace rancher_corn_cost_l1871_187165

/-- Represents the rancher's situation --/
structure RancherSituation where
  sheep : Nat
  cattle : Nat
  grassAcres : Nat
  grassPerCowPerMonth : Nat
  grassPerSheepPerMonth : Nat
  monthsPerBagForCow : Nat
  monthsPerBagForSheep : Nat
  cornBagPrice : Nat

/-- Calculates the yearly cost of feed corn for the rancher --/
def yearlyCornCost (s : RancherSituation) : Nat :=
  let totalGrassPerMonth := s.cattle * s.grassPerCowPerMonth + s.sheep * s.grassPerSheepPerMonth
  let grazingMonths := s.grassAcres / totalGrassPerMonth
  let cornMonths := 12 - grazingMonths
  let cornForSheep := (cornMonths * s.sheep + s.monthsPerBagForSheep - 1) / s.monthsPerBagForSheep
  let cornForCattle := cornMonths * s.cattle / s.monthsPerBagForCow
  (cornForSheep + cornForCattle) * s.cornBagPrice

/-- Theorem stating that the rancher needs to spend $360 on feed corn each year --/
theorem rancher_corn_cost :
  let s : RancherSituation := {
    sheep := 8,
    cattle := 5,
    grassAcres := 144,
    grassPerCowPerMonth := 2,
    grassPerSheepPerMonth := 1,
    monthsPerBagForCow := 1,
    monthsPerBagForSheep := 2,
    cornBagPrice := 10
  }
  yearlyCornCost s = 360 := by sorry

end rancher_corn_cost_l1871_187165


namespace gcd_problem_l1871_187119

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 1729 * k) : 
  Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := by
sorry

end gcd_problem_l1871_187119


namespace simple_random_for_small_population_l1871_187179

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Other

/-- Determines the appropriate sampling method based on population size -/
def appropriateSamplingMethod (populationSize : ℕ) (sampleSize : ℕ) : SamplingMethod :=
  if populationSize ≤ 10 ∧ sampleSize = 1 then
    SamplingMethod.SimpleRandom
  else
    SamplingMethod.Other

/-- Theorem: For a population of 10 items with 1 item randomly selected,
    the appropriate sampling method is simple random sampling -/
theorem simple_random_for_small_population :
  appropriateSamplingMethod 10 1 = SamplingMethod.SimpleRandom :=
by sorry

end simple_random_for_small_population_l1871_187179


namespace remainder_of_prime_powers_l1871_187171

theorem remainder_of_prime_powers (p q : Nat) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q - 1) + q^(p - 1)) % (p * q) = 1 := by
  sorry

end remainder_of_prime_powers_l1871_187171


namespace not_sufficient_not_necessary_l1871_187108

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem not_sufficient_not_necessary 
  (m : Line) (α β : Plane) 
  (h_perp_planes : perpendicular_planes α β) :
  ¬(∀ m α β, parallel m α → perpendicular m β) ∧ 
  ¬(∀ m α β, perpendicular m β → parallel m α) :=
sorry

end not_sufficient_not_necessary_l1871_187108


namespace parallelogram_area_34_18_l1871_187186

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 34 cm and height 18 cm is 612 square centimeters -/
theorem parallelogram_area_34_18 : parallelogram_area 34 18 = 612 := by
  sorry

end parallelogram_area_34_18_l1871_187186


namespace unique_coprime_squares_l1871_187196

theorem unique_coprime_squares (m n : ℕ+) : 
  m.val.Coprime n.val ∧ 
  ∃ x y : ℕ, (m.val^2 - 5*n.val^2 = x^2) ∧ (m.val^2 + 5*n.val^2 = y^2) →
  m.val = 41 ∧ n.val = 12 :=
by sorry

end unique_coprime_squares_l1871_187196


namespace polynomial_remainder_l1871_187103

theorem polynomial_remainder (x : ℝ) : 
  let p : ℝ → ℝ := λ x => 5*x^8 - 3*x^7 + 2*x^6 - 4*x^3 + x^2 - 9
  let d : ℝ → ℝ := λ x => 3*x - 9
  ∃ q : ℝ → ℝ, p = λ x => d x * q x + 39594 := by
  sorry

end polynomial_remainder_l1871_187103


namespace expression_simplification_l1871_187197

theorem expression_simplification (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 := by
  sorry

end expression_simplification_l1871_187197


namespace min_value_ab_l1871_187142

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 4/b = Real.sqrt (a*b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 4/y = Real.sqrt (x*y) → a*b ≤ x*y :=
by sorry

end min_value_ab_l1871_187142


namespace simplify_and_evaluate_l1871_187184

theorem simplify_and_evaluate : 
  let x : ℚ := 1/2
  5 * x^2 - (x^2 - 2*(2*x - 3)) = -3 := by sorry

end simplify_and_evaluate_l1871_187184


namespace fourth_term_is_five_l1871_187191

/-- An arithmetic sequence where the sum of the third and fifth terms is 10 -/
def ArithmeticSequence (a : ℝ) (d : ℝ) : Prop :=
  a + (a + 2*d) = 10

/-- The fourth term of the arithmetic sequence -/
def FourthTerm (a : ℝ) (d : ℝ) : ℝ := a + d

theorem fourth_term_is_five {a d : ℝ} (h : ArithmeticSequence a d) : FourthTerm a d = 5 := by
  sorry

end fourth_term_is_five_l1871_187191


namespace modified_triangle_invalid_zero_area_l1871_187172

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given side lengths can form a valid triangle -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- The original triangle ABC -/
def originalTriangle : Triangle :=
  { a := 12, b := 7, c := 10 }

/-- The modified triangle with doubled AB and AC -/
def modifiedTriangle : Triangle :=
  { a := 24, b := 14, c := 10 }

/-- Theorem stating that the modified triangle is not valid and has zero area -/
theorem modified_triangle_invalid_zero_area :
  ¬(isValidTriangle modifiedTriangle) ∧ 
  (∃ area : ℝ, area = 0 ∧ area ≥ 0) :=
by sorry

end modified_triangle_invalid_zero_area_l1871_187172


namespace area_between_curves_l1871_187198

theorem area_between_curves : 
  let f (x : ℝ) := Real.exp x
  let g (x : ℝ) := Real.exp (-x)
  let a := 0
  let b := 1
  ∫ x in a..b, (f x - g x) = Real.exp 1 + Real.exp (-1) - 2 := by
  sorry

end area_between_curves_l1871_187198


namespace marble_bag_total_l1871_187194

/-- Given a bag of marbles with red:blue:green ratio of 2:4:5 and 40 blue marbles,
    the total number of marbles is 110. -/
theorem marble_bag_total (red blue green total : ℕ) : 
  red + blue + green = total →
  red = 2 * n ∧ blue = 4 * n ∧ green = 5 * n →
  blue = 40 →
  total = 110 := by
  sorry

end marble_bag_total_l1871_187194


namespace sqrt_division_minus_abs_l1871_187185

theorem sqrt_division_minus_abs : Real.sqrt 63 / Real.sqrt 7 - |(-4)| = -1 := by
  sorry

end sqrt_division_minus_abs_l1871_187185


namespace imaginary_part_of_z_l1871_187129

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) :
  z.im = -1 := by sorry

end imaginary_part_of_z_l1871_187129


namespace expression_simplification_l1871_187123

theorem expression_simplification (m : ℝ) (h : m^2 - 2*m - 1 = 0) :
  (m + 2) / (2*m^2 - 6*m) / (m + 3 + 5 / (m - 3)) = 1/2 := by
  sorry

end expression_simplification_l1871_187123


namespace BC_time_proof_l1871_187128

-- Define work rates for A, B, and C
def A_rate : ℚ := 1 / 4
def B_rate : ℚ := 1 / 12
def AC_rate : ℚ := 1 / 2

-- Define the time taken by B and C together
def BC_time : ℚ := 3

-- Theorem statement
theorem BC_time_proof :
  let C_rate : ℚ := AC_rate - A_rate
  let BC_rate : ℚ := B_rate + C_rate
  BC_time = 1 / BC_rate :=
by sorry

end BC_time_proof_l1871_187128


namespace machinery_expenditure_l1871_187115

/-- Proves that the amount spent on machinery is $2000 --/
theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 5555.56 →
  raw_materials = 3000 →
  cash_percentage = 0.1 →
  total = raw_materials + (total * cash_percentage) + 2000 := by
  sorry

end machinery_expenditure_l1871_187115


namespace alcohol_mixture_problem_l1871_187134

theorem alcohol_mixture_problem (x : ℝ) :
  (x * 50 + 30 * 150) / (50 + 150) = 25 → x = 10 := by
  sorry

end alcohol_mixture_problem_l1871_187134


namespace perimeter_of_rearranged_rectangles_l1871_187111

/-- The perimeter of a shape formed by rearranging two equal rectangles cut from a square --/
theorem perimeter_of_rearranged_rectangles (square_side : ℝ) : square_side = 100 → 500 = 3 * square_side + 4 * (square_side / 2) := by
  sorry

end perimeter_of_rearranged_rectangles_l1871_187111


namespace shortest_player_height_l1871_187136

theorem shortest_player_height (tallest_height shortest_height height_difference : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5)
  (h3 : tallest_height = shortest_height + height_difference) :
  shortest_height = 68.25 := by
  sorry

end shortest_player_height_l1871_187136


namespace average_speed_calculation_l1871_187118

/-- Calculates the average speed of a trip given the following conditions:
  * The trip lasts for 12 hours
  * The car travels at 45 mph for the first 4 hours
  * The car travels at 75 mph for the remaining hours
-/
theorem average_speed_calculation (total_time : ℝ) (initial_speed : ℝ) (initial_duration : ℝ) (final_speed : ℝ) :
  total_time = 12 →
  initial_speed = 45 →
  initial_duration = 4 →
  final_speed = 75 →
  (initial_speed * initial_duration + final_speed * (total_time - initial_duration)) / total_time = 65 := by
  sorry

#check average_speed_calculation

end average_speed_calculation_l1871_187118


namespace intersection_M_N_l1871_187157

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end intersection_M_N_l1871_187157


namespace max_value_of_f_l1871_187110

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ f 0 ∧ f 0 = 3) ∨
  (a > 2 ∧ ∀ x ∈ Set.Icc 0 a, f x ≤ f a ∧ f a = a^2 - 2*a + 3) :=
sorry

end max_value_of_f_l1871_187110


namespace highest_score_is_179_l1871_187141

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  averageScore : ℝ
  highestScore : ℝ
  lowestScore : ℝ
  averageExcludingExtremes : ℝ

/-- Theorem: Given the batsman's statistics, prove that the highest score is 179 runs -/
theorem highest_score_is_179 (stats : BatsmanStats)
  (h1 : stats.totalInnings = 46)
  (h2 : stats.averageScore = 60)
  (h3 : stats.highestScore - stats.lowestScore = 150)
  (h4 : stats.averageExcludingExtremes = 58) :
  stats.highestScore = 179 := by
  sorry

#check highest_score_is_179

end highest_score_is_179_l1871_187141


namespace tangent_length_to_circle_l1871_187143

/-- The length of the tangent from a point to a circle -/
theorem tangent_length_to_circle (x y : ℝ) : 
  let p : ℝ × ℝ := (2, 3)
  let center : ℝ × ℝ := (1, 1)
  let radius : ℝ := 1
  let dist_squared : ℝ := (p.1 - center.1)^2 + (p.2 - center.2)^2
  (x - 1)^2 + (y - 1)^2 = 1 →  -- Circle equation
  dist_squared > radius^2 →    -- P is outside the circle
  Real.sqrt (dist_squared - radius^2) = 2 := by
sorry

end tangent_length_to_circle_l1871_187143


namespace mathland_license_plate_probability_l1871_187159

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of possible two-digit numbers --/
def two_digit_numbers : ℕ := 100

/-- The total number of possible license plates in Mathland --/
def total_license_plates : ℕ := alphabet_size * (alphabet_size - 1) * (alphabet_size - 2) * two_digit_numbers

/-- The probability of a specific license plate configuration in Mathland --/
def specific_plate_probability : ℚ := 1 / total_license_plates

theorem mathland_license_plate_probability :
  specific_plate_probability = 1 / 1560000 := by sorry

end mathland_license_plate_probability_l1871_187159


namespace second_smallest_natural_with_remainder_l1871_187117

theorem second_smallest_natural_with_remainder : ∃ n : ℕ, 
  n > 500 ∧ 
  n % 7 = 3 ∧ 
  (∃! m : ℕ, m > 500 ∧ m % 7 = 3 ∧ m < n) ∧
  n = 514 :=
by sorry

end second_smallest_natural_with_remainder_l1871_187117


namespace no_real_roots_iff_k_gt_two_l1871_187163

theorem no_real_roots_iff_k_gt_two (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * x + (1/2) ≠ 0) ↔ k > 2 := by
  sorry

end no_real_roots_iff_k_gt_two_l1871_187163


namespace cindys_calculation_l1871_187137

theorem cindys_calculation (x : ℤ) (h : (x - 7) / 5 = 51) : (x - 5) / 7 = 36 := by
  sorry

end cindys_calculation_l1871_187137


namespace committee_formation_count_l1871_187122

def club_size : ℕ := 12
def committee_size : ℕ := 5
def president_count : ℕ := 1

theorem committee_formation_count :
  (club_size.choose committee_size) - ((club_size - president_count).choose committee_size) = 330 :=
by sorry

end committee_formation_count_l1871_187122


namespace roses_picked_l1871_187183

theorem roses_picked (initial : ℕ) (sold : ℕ) (final : ℕ) 
  (h1 : initial = 37) 
  (h2 : sold = 16) 
  (h3 : final = 40) : 
  final - (initial - sold) = 19 := by
sorry

end roses_picked_l1871_187183


namespace yellow_crayon_count_l1871_187190

/-- Given the number of red, blue, and yellow crayons with specific relationships,
    prove that the number of yellow crayons is 32. -/
theorem yellow_crayon_count :
  ∀ (red blue yellow : ℕ),
  red = 14 →
  blue = red + 5 →
  yellow = 2 * blue - 6 →
  yellow = 32 := by
sorry

end yellow_crayon_count_l1871_187190


namespace triangle_square_perimeter_difference_l1871_187166

theorem triangle_square_perimeter_difference (d : ℤ) : 
  (∃ (t s : ℝ), 3 * t - 4 * s = 1575 ∧ t - s = d ∧ s > 0) ↔ d > 525 :=
sorry

end triangle_square_perimeter_difference_l1871_187166


namespace percentage_of_flowering_plants_l1871_187175

/-- Proves that the percentage of flowering plants is 40% given the conditions --/
theorem percentage_of_flowering_plants 
  (total_plants : ℕ)
  (porch_fraction : ℚ)
  (flowers_per_plant : ℕ)
  (total_porch_flowers : ℕ)
  (h1 : total_plants = 80)
  (h2 : porch_fraction = 1 / 4)
  (h3 : flowers_per_plant = 5)
  (h4 : total_porch_flowers = 40) :
  (total_porch_flowers : ℚ) / (porch_fraction * flowers_per_plant * total_plants) = 40 / 100 :=
by sorry

end percentage_of_flowering_plants_l1871_187175


namespace min_square_value_l1871_187113

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ x : ℕ+, (15 * a.val + 16 * b.val : ℕ) = x * x)
  (h2 : ∃ y : ℕ+, (16 * a.val - 15 * b.val : ℕ) = y * y) :
  min (15 * a.val + 16 * b.val) (16 * a.val - 15 * b.val) ≥ 231361 :=
by sorry

end min_square_value_l1871_187113


namespace product_value_l1871_187188

theorem product_value (x : ℝ) (h : Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8) :
  (6 + x) * (21 - x) = 1369 / 4 := by
  sorry

end product_value_l1871_187188


namespace hide_and_seek_l1871_187102

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem to prove
theorem hide_and_seek :
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena :=
sorry

end hide_and_seek_l1871_187102


namespace common_chord_length_l1871_187109

theorem common_chord_length (r : ℝ) (h : r = 12) :
  let chord_length := 2 * (r * Real.sqrt 3)
  chord_length = 12 * Real.sqrt 3 := by sorry

end common_chord_length_l1871_187109


namespace complex_fraction_evaluation_l1871_187187

theorem complex_fraction_evaluation : 
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end complex_fraction_evaluation_l1871_187187


namespace simplify_expression_l1871_187192

theorem simplify_expression (m : ℝ) (h : m < 1) :
  (m - 1) * Real.sqrt (-1 / (m - 1)) = -Real.sqrt (1 - m) := by
  sorry

end simplify_expression_l1871_187192


namespace reciprocal_problem_l1871_187152

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 := by
  sorry

end reciprocal_problem_l1871_187152


namespace product_closest_to_105_l1871_187160

def product : ℝ := 2.1 * (50.2 + 0.09)

def options : List ℝ := [100, 105, 106, 110]

theorem product_closest_to_105 : 
  ∀ x ∈ options, |product - 105| ≤ |product - x| := by
  sorry

end product_closest_to_105_l1871_187160


namespace tripod_height_after_damage_l1871_187126

/-- Represents the height of a tripod after one leg is shortened -/
def tripod_height (leg_length : ℝ) (initial_height : ℝ) (shortened_length : ℝ) : ℝ :=
  -- Define the function to calculate the new height
  sorry

theorem tripod_height_after_damage :
  let leg_length : ℝ := 6
  let initial_height : ℝ := 5
  let shortened_length : ℝ := 1
  tripod_height leg_length initial_height shortened_length = 5 := by
  sorry

#check tripod_height_after_damage

end tripod_height_after_damage_l1871_187126


namespace code_cracking_probabilities_l1871_187130

/-- The probability of person i cracking the code -/
def P (i : Fin 3) : ℚ :=
  match i with
  | 0 => 1/5
  | 1 => 1/4
  | 2 => 1/3

/-- The probability that exactly two people crack the code -/
def prob_two_crack : ℚ :=
  P 0 * P 1 * (1 - P 2) + P 0 * (1 - P 1) * P 2 + (1 - P 0) * P 1 * P 2

/-- The probability that no one cracks the code -/
def prob_none_crack : ℚ :=
  (1 - P 0) * (1 - P 1) * (1 - P 2)

theorem code_cracking_probabilities :
  prob_two_crack = 3/20 ∧ 
  (1 - prob_none_crack) > prob_none_crack := by
  sorry


end code_cracking_probabilities_l1871_187130


namespace triangle_perimeter_l1871_187168

/-- An ellipse with equation x²/a² + y²/9 = 1, where a > 3 -/
structure Ellipse where
  a : ℝ
  h_a : a > 3

/-- The foci of the ellipse -/
structure Foci (e : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_dist : dist F₁ F₂ = 8

/-- A chord AB passing through F₁ -/
structure Chord (e : Ellipse) (f : Foci e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_pass : A.1 = f.F₁.1 ∧ A.2 = f.F₁.2

/-- The theorem stating that the perimeter of triangle ABF₂ is 20 -/
theorem triangle_perimeter (e : Ellipse) (f : Foci e) (c : Chord e f) :
  dist c.A c.B + dist c.B f.F₂ + dist c.A f.F₂ = 20 := by
  sorry

end triangle_perimeter_l1871_187168


namespace trig_identity_l1871_187112

theorem trig_identity : 
  Real.sin (40 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (220 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end trig_identity_l1871_187112


namespace role_assignment_count_l1871_187162

/-- The number of ways to assign roles in a play. -/
def assign_roles (num_men : ℕ) (num_women : ℕ) (male_roles : ℕ) (female_roles : ℕ) (either_roles : ℕ) : ℕ :=
  (num_men.descFactorial male_roles) *
  (num_women.descFactorial female_roles) *
  ((num_men + num_women - male_roles - female_roles).descFactorial either_roles)

/-- Theorem stating the number of ways to assign roles in the given scenario. -/
theorem role_assignment_count :
  assign_roles 7 8 3 3 4 = 213955200 :=
by sorry

end role_assignment_count_l1871_187162


namespace ratio_equality_l1871_187127

theorem ratio_equality (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) :
  (a / 8) / (b / 7) = 1 := by
  sorry

end ratio_equality_l1871_187127


namespace jerrys_breakfast_calories_l1871_187170

/-- Represents the number of pancakes in Jerry's breakfast -/
def num_pancakes : ℕ := 6

/-- Represents the calories per pancake -/
def calories_per_pancake : ℕ := 120

/-- Represents the number of bacon strips in Jerry's breakfast -/
def num_bacon_strips : ℕ := 2

/-- Represents the calories per bacon strip -/
def calories_per_bacon_strip : ℕ := 100

/-- Represents the calories in the bowl of cereal -/
def cereal_calories : ℕ := 200

/-- Theorem stating that the total calories in Jerry's breakfast is 1120 -/
theorem jerrys_breakfast_calories : 
  num_pancakes * calories_per_pancake + 
  num_bacon_strips * calories_per_bacon_strip + 
  cereal_calories = 1120 := by
  sorry

end jerrys_breakfast_calories_l1871_187170


namespace power_equality_l1871_187169

theorem power_equality (p : ℕ) : 81^6 = 3^p → p = 24 := by
  sorry

end power_equality_l1871_187169


namespace min_cost_theorem_min_cost_value_l1871_187148

def volleyball_price : ℕ := 50
def basketball_price : ℕ := 80

def total_balls : ℕ := 60
def max_cost : ℕ := 3800
def max_volleyballs : ℕ := 38

def cost_function (m : ℕ) : ℕ := volleyball_price * m + basketball_price * (total_balls - m)

theorem min_cost_theorem (m : ℕ) (h1 : m ≤ max_volleyballs) (h2 : cost_function m ≤ max_cost) :
  cost_function max_volleyballs ≤ cost_function m :=
sorry

theorem min_cost_value : cost_function max_volleyballs = 3660 :=
sorry

end min_cost_theorem_min_cost_value_l1871_187148


namespace water_in_bucket_A_l1871_187151

/-- Given two buckets A and B, prove that the original amount of water in bucket A is 20 kg. -/
theorem water_in_bucket_A (bucket_A bucket_B : ℝ) : 
  (0.2 * bucket_A = 0.4 * bucket_B) → 
  (0.6 * bucket_B = 6) → 
  bucket_A = 20 := by sorry

end water_in_bucket_A_l1871_187151


namespace sum_max_min_f_l1871_187195

def f (x : ℝ) := -x^2 + 2*x + 3

theorem sum_max_min_f : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 4 :=
by
  sorry


end sum_max_min_f_l1871_187195


namespace rope_length_ratio_l1871_187180

def joeys_rope_length : ℕ := 56
def chads_rope_length : ℕ := 21

theorem rope_length_ratio : 
  (joeys_rope_length : ℚ) / (chads_rope_length : ℚ) = 8 / 3 := by
  sorry

end rope_length_ratio_l1871_187180


namespace cyclic_sum_inequality_l1871_187133

theorem cyclic_sum_inequality (x y z : ℝ) (α : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) (hα : α ≥ 0) :
  ((x^(α+3) + y^(α+3)) / (x^2 + x*y + y^2) +
   (y^(α+3) + z^(α+3)) / (y^2 + y*z + z^2) +
   (z^(α+3) + x^(α+3)) / (z^2 + z*x + x^2)) ≥ 2 := by
sorry

end cyclic_sum_inequality_l1871_187133


namespace functional_sequence_a10_l1871_187106

/-- A sequence satisfying a functional equation -/
def FunctionalSequence (a : ℕ+ → ℤ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p + a q

theorem functional_sequence_a10 (a : ℕ+ → ℤ) 
  (h1 : FunctionalSequence a) (h2 : a 2 = -6) : 
  a 10 = -30 := by sorry

end functional_sequence_a10_l1871_187106


namespace john_leftover_percentage_l1871_187101

/-- The percentage of earnings John spent on rent -/
def rent_percentage : ℝ := 40

/-- The percentage less than rent that John spent on the dishwasher -/
def dishwasher_percentage_less : ℝ := 30

/-- Theorem stating that the percentage of John's earnings left over is 48% -/
theorem john_leftover_percentage : 
  100 - (rent_percentage + (100 - dishwasher_percentage_less) / 100 * rent_percentage) = 48 := by
  sorry

end john_leftover_percentage_l1871_187101


namespace symmetry_of_point_l1871_187135

def symmetric_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem symmetry_of_point :
  symmetric_point_x_axis (1, 2) = (1, -2) := by
sorry

end symmetry_of_point_l1871_187135


namespace unique_score_above_90_l1871_187125

/-- Represents the scoring system for the exam -/
structure ScoringSystem where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ

/-- Calculates the score given the number of correct and incorrect answers -/
def calculate_score (system : ScoringSystem) (correct : ℕ) (incorrect : ℕ) : ℤ :=
  system.correct_points * correct + system.incorrect_points * incorrect

/-- Checks if a score uniquely determines the number of correct and incorrect answers -/
def is_unique_score (system : ScoringSystem) (score : ℤ) : Prop :=
  ∃! (correct incorrect : ℕ),
    correct + incorrect ≤ system.total_questions ∧
    calculate_score system correct incorrect = score

/-- The main theorem to prove -/
theorem unique_score_above_90 (system : ScoringSystem) : 
  system.total_questions = 35 →
  system.correct_points = 5 →
  system.incorrect_points = -2 →
  (∀ s, s > 90 ∧ s < 116 → ¬is_unique_score system s) →
  is_unique_score system 116 := 
by sorry

end unique_score_above_90_l1871_187125


namespace solve_system_for_q_l1871_187164

theorem solve_system_for_q : 
  ∀ p q : ℚ, 3 * p + 4 * q = 8 → 4 * p + 3 * q = 13 → q = -1 := by
  sorry

end solve_system_for_q_l1871_187164


namespace x0_value_l1871_187161

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2) → x₀ = exp 1 := by
  sorry

end x0_value_l1871_187161


namespace greatest_integer_not_divisible_by_1111_l1871_187131

theorem greatest_integer_not_divisible_by_1111 :
  (∃ (N : ℕ), N > 0 ∧
    (∃ (x : Fin N → ℤ), ∀ (i j : Fin N), i ≠ j →
      ¬(1111 ∣ (x i)^2 - (x i) * (x j))) ∧
    (∀ (M : ℕ), M > N →
      ¬(∃ (y : Fin M → ℤ), ∀ (i j : Fin M), i ≠ j →
        ¬(1111 ∣ (y i)^2 - (y i) * (y j)))) ∧
  N = 1000) :=
sorry

end greatest_integer_not_divisible_by_1111_l1871_187131


namespace dormitory_problem_l1871_187167

theorem dormitory_problem : ∃! x : ℕ+, ∃ n : ℕ+, 
  (x = 4 * n + 20) ∧ 
  (↑(n - 1) < (↑x : ℚ) / 8 ∧ (↑x : ℚ) / 8 < ↑n) := by
  sorry

end dormitory_problem_l1871_187167


namespace no_perfect_square_n_n_plus_one_l1871_187155

theorem no_perfect_square_n_n_plus_one (n : ℕ) (hn : n > 0) : 
  ¬∃ (k : ℕ), n * (n + 1) = k^2 := by
  sorry

end no_perfect_square_n_n_plus_one_l1871_187155


namespace sphere_surface_area_with_holes_value_l1871_187176

/-- The surface area of a sphere with diameter 10 inches, after drilling three holes each with a radius of 0.5 inches -/
def sphere_surface_area_with_holes : ℝ := sorry

/-- The diameter of the bowling ball in inches -/
def ball_diameter : ℝ := 10

/-- The number of finger holes -/
def num_holes : ℕ := 3

/-- The radius of each finger hole in inches -/
def hole_radius : ℝ := 0.5

theorem sphere_surface_area_with_holes_value :
  sphere_surface_area_with_holes = (197 / 2) * Real.pi := by sorry

end sphere_surface_area_with_holes_value_l1871_187176


namespace sum_first_150_remainder_l1871_187154

theorem sum_first_150_remainder (n : Nat) (h : n = 150) :
  (List.range n).sum % 8000 = 3325 := by
  sorry

end sum_first_150_remainder_l1871_187154


namespace square_of_one_plus_i_l1871_187147

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem square_of_one_plus_i : (1 + i)^2 = 2*i := by sorry

end square_of_one_plus_i_l1871_187147


namespace extremum_point_of_f_l1871_187132

def f (x : ℝ) := x^2 - 2*x

theorem extremum_point_of_f :
  ∃ (c : ℝ), c = 1 ∧ ∀ (x : ℝ), f x ≤ f c ∨ f x ≥ f c :=
sorry

end extremum_point_of_f_l1871_187132
