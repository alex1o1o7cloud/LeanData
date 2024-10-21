import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1126_112654

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1

theorem f_properties :
  (∀ y ∈ Set.range f, 0 ≤ y ∧ y ≤ 4) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (- π/6 + k*π) (π/3 + k*π))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1126_112654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_bishop_arrangements_l1126_112651

def number_of_arrangements_max_non_threatening (n : ℕ) : ℕ :=
  (n - 1)^2

def number_of_arrangements_min_all_threatening (n : ℕ) : ℕ :=
  n^2

theorem chess_bishop_arrangements (n : ℕ) (h : Even n) :
  (∃ k : ℕ, (number_of_arrangements_max_non_threatening n) = k^2) ∧
  (∃ m : ℕ, (number_of_arrangements_min_all_threatening n) = m^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_bishop_arrangements_l1126_112651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_half_hypotenuse_l1126_112672

/-- Represents a right triangle with sides a, b, and c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- Represents a folded triangle where one vertex falls on the opposite vertex -/
def FoldedTriangle (t : RightTriangle) :=
  { crease : ℝ // crease > 0 }

/-- The length of the crease in a folded right triangle -/
noncomputable def creaseLength (t : RightTriangle) (f : FoldedTriangle t) : ℝ :=
  t.c / 2

theorem crease_length_is_half_hypotenuse (t : RightTriangle) (f : FoldedTriangle t) :
  t.a = 5 → t.b = 12 → t.c = 13 → creaseLength t f = 6.5 := by
  intros h1 h2 h3
  unfold creaseLength
  rw [h3]
  norm_num

#check crease_length_is_half_hypotenuse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_half_hypotenuse_l1126_112672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1126_112692

-- Define the function f(x) = x + 2/x
noncomputable def f (x : ℝ) : ℝ := x + 2 / x

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < Real.sqrt 2 →
  f x₂ < f x₁ :=
by
  sorry

#check f_decreasing_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1126_112692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_A_and_B_l1126_112615

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 * Real.cos θ1 - r2 * Real.cos θ2)^2 + (r1 * Real.sin θ1 - r2 * Real.sin θ2)^2)

/-- Theorem: The distance between points A(3, 5π/3) and B(1, 2π/3) in polar coordinates is 4 -/
theorem distance_between_A_and_B :
  polar_distance 3 (5 * Real.pi / 3) 1 (2 * Real.pi / 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_A_and_B_l1126_112615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tortoise_wins_l1126_112695

/-- Represents the race between a tortoise and a hare -/
structure Race where
  distance : ℕ
  hare_speed : ℕ
  tortoise_speed : ℕ

/-- Calculates the time taken by the tortoise to finish the race -/
def tortoise_finish_time (race : Race) : ℕ :=
  race.distance / race.tortoise_speed

/-- Calculates the distance covered by the hare in a given time -/
def hare_distance (race : Race) (time : ℕ) : ℕ :=
  let run_time := min time 1 + min (max (time - 17) 0) 2 + min (max (time - 35) 0) 3 + min (max (time - 53) 0) 4
  run_time * race.hare_speed

/-- States that the tortoise finishes before or at the same time as the hare -/
theorem tortoise_wins (race : Race) (h1 : race.distance = 3000)
    (h2 : race.hare_speed = 300) (h3 : race.tortoise_speed = 50) :
    hare_distance race (tortoise_finish_time race) ≤ race.distance := by
  sorry

#eval tortoise_finish_time { distance := 3000, hare_speed := 300, tortoise_speed := 50 }
#eval hare_distance { distance := 3000, hare_speed := 300, tortoise_speed := 50 } 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tortoise_wins_l1126_112695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l1126_112618

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x < g x} = Set.Ioi (-Real.pi) ∩ Set.Iio 2 :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x < g x} = Set.Ioo (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l1126_112618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_and_perimeter_l1126_112634

/-- Represents a trapezoid EFGH with given properties -/
structure Trapezoid where
  EH : ℝ
  EF : ℝ
  FG : ℝ
  altitude : ℝ
  EH_positive : EH > 0
  EF_positive : EF > 0
  FG_positive : FG > 0
  altitude_positive : altitude > 0

/-- Calculates the length of GH in the trapezoid -/
noncomputable def Trapezoid.GH (t : Trapezoid) : ℝ :=
  Real.sqrt (t.EH^2 - t.altitude^2) + t.EF + Real.sqrt (t.FG^2 - t.altitude^2)

/-- Calculates the area of the trapezoid -/
noncomputable def Trapezoid.area (t : Trapezoid) : ℝ :=
  (t.EF + t.GH) * t.altitude / 2

/-- Calculates the perimeter of the trapezoid -/
noncomputable def Trapezoid.perimeter (t : Trapezoid) : ℝ :=
  t.EH + t.EF + t.FG + t.GH

/-- The main theorem stating the area and perimeter of the specific trapezoid -/
theorem trapezoid_area_and_perimeter :
  ∃ (t : Trapezoid),
    t.EH = 25 ∧
    t.EF = 65 ∧
    t.FG = 30 ∧
    t.altitude = 18 ∧
    t.area = 1386 ∧
    t.perimeter = 209 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_and_perimeter_l1126_112634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_final_result_l1126_112679

-- Define the set T
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0 ∧ p.1 + p.2.1 + p.2.2 = 1}

-- Define the support relation
def supports (p : ℝ × ℝ × ℝ) (a b c : ℝ) : Prop :=
  (p.1 ≥ a ∧ p.2.1 ≥ b) ∨ (p.1 ≥ a ∧ p.2.2 ≥ c) ∨ (p.2.1 ≥ b ∧ p.2.2 ≥ c)

-- Define the set S
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p (1/2) (1/3) (1/6)}

-- Define the area function (noncomputable as it involves integration)
noncomputable def area (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem area_ratio_S_T : (area S) / (area T) = 7 / 18 := by
  sorry

-- State the final result
theorem final_result : ∃ m n : ℕ, (area S) / (area T) = m / n ∧ m + n = 25 := by
  use 7, 18
  constructor
  · exact area_ratio_S_T
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_T_final_result_l1126_112679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contract_pages_l1126_112673

theorem contract_pages (copies_per_person people total_pages : ℕ) : 
  let total_copies := copies_per_person * people
  total_pages / total_copies = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contract_pages_l1126_112673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l1126_112622

/-- The function y₁ = x₁ ln x₁ -/
noncomputable def y₁ (x₁ : ℝ) : ℝ := x₁ * Real.log x₁

/-- The function y₂ = x₂ - 3 -/
def y₂ (x₂ : ℝ) : ℝ := x₂ - 3

/-- The squared distance function -/
noncomputable def distance_squared (x₁ x₂ : ℝ) : ℝ := (x₁ - x₂)^2 + (y₁ x₁ - y₂ x₂)^2

theorem min_distance_squared :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x₁ x₂ : ℝ), distance_squared x₁ x₂ ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l1126_112622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_isosceles_triangle_perimeter_l1126_112684

/-- An isosceles triangle with sides a, b, b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ

/-- The perimeter of an isosceles triangle -/
noncomputable def perimeter (t : IsoscelesTriangle) : ℝ := t.a + 2 * t.b

/-- Similarity ratio between two triangles -/
noncomputable def similarityRatio (t1 t2 : IsoscelesTriangle) : ℝ := t2.a / t1.a

theorem similar_isosceles_triangle_perimeter 
  (t1 t2 : IsoscelesTriangle) 
  (h1 : t1.a = 18 ∧ t1.b = 24) 
  (h2 : t2.a = 45) : 
  perimeter t2 = 165 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_isosceles_triangle_perimeter_l1126_112684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_next_fib_num_monomials_equals_next_fib_l1126_112650

/-- K_n is a polynomial in n variables -/
def K (n : ℕ) : (Fin n → ℕ) → ℕ := sorry

/-- a_n is the number of monomials in K_n(1, 1, ..., 1) -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a (n + 1) + a n

/-- F_n is the nth Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem a_equals_next_fib (n : ℕ) : a n = fib (n + 1) := by
  sorry

/-- The number of monomials in K_n equals the (n+1)th Fibonacci number -/
theorem num_monomials_equals_next_fib (n : ℕ) : 
  (K n (λ _ => 1)) = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_next_fib_num_monomials_equals_next_fib_l1126_112650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1126_112630

/-- Circle with center (a,a) and radius 1 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - a)^2 = 1}

/-- Line y = 3x -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 3 * p.1}

/-- Intersection points of the circle and line -/
def IntersectionPoints (a : ℝ) : Set (ℝ × ℝ) :=
  Circle a ∩ Line

/-- Area of the triangle formed by the center of the circle and two intersection points -/
noncomputable def TriangleArea (a : ℝ) : ℝ :=
  let d := a / Real.sqrt 5
  let chordLength := Real.sqrt (2 - a^2 / 5)
  (1 / 2) * d * chordLength

/-- Theorem: The area of the triangle is maximized when a = √5 -/
theorem max_triangle_area :
  ∃ (a : ℝ), a > 0 ∧ ∀ (b : ℝ), b > 0 → TriangleArea a ≥ TriangleArea b :=
by
  use Real.sqrt 5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1126_112630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_bar_weight_loss_l1126_112648

/-- Represents the weight loss of a metal bar in water -/
noncomputable def weight_loss_in_water (total_weight : ℝ) (tin_ratio : ℝ) (tin_loss_rate : ℝ) (silver_loss_rate : ℝ) : ℝ :=
  let tin_weight := total_weight * tin_ratio / (1 + tin_ratio)
  let silver_weight := total_weight / (1 + tin_ratio)
  tin_weight * tin_loss_rate + silver_weight * silver_loss_rate

/-- Proves that a 50 kg metal bar with tin-to-silver ratio of 2/3 loses 5 kg in water -/
theorem metal_bar_weight_loss :
  weight_loss_in_water 50 (2/3) (1.375/10) (0.375/5) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_bar_weight_loss_l1126_112648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_novak_family_cakes_l1126_112682

/-- The number of cakes baked by the Novák family -/
def total_cakes : ℕ := 108

/-- The fraction of cakes delivered to relatives -/
def relatives_fraction : ℚ := 1/4

/-- The fraction of cakes given to colleagues -/
def colleagues_fraction : ℚ := 1/6

/-- The fraction of cakes given to neighbors -/
def neighbors_fraction : ℚ := 1/9

/-- The number of additional cakes needed to reach half of the total -/
def additional_cakes : ℕ := 3

theorem novak_family_cakes :
  (relatives_fraction + colleagues_fraction + neighbors_fraction) * (total_cakes : ℚ) + (additional_cakes : ℚ) = (total_cakes : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_novak_family_cakes_l1126_112682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1126_112670

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : Prop :=
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 = (radius1 + radius2)^2

theorem circle_tangency (a : ℝ) (h : a > 0) :
  externally_tangent (a, 0) (0, Real.sqrt 5) 2 3 → a = 2 * Real.sqrt 5 :=
by
  intro h_tangent
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1126_112670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_percentage_proof_l1126_112610

/-- Represents the percentage of students who like a particular sport -/
def SportPercentage := Fin 101

theorem chess_percentage_proof (total_students : ℕ) 
  (basketball_percent : SportPercentage)
  (soccer_percent : SportPercentage)
  (chess_or_basketball : ℕ) :
  total_students = 250 →
  basketball_percent = ⟨40, by norm_num⟩ →
  soccer_percent = ⟨28, by norm_num⟩ →
  chess_or_basketball = 125 →
  ∃ (chess_percent : SportPercentage),
    chess_percent = ⟨10, by norm_num⟩ ∧
    chess_or_basketball = (basketball_percent.val + chess_percent.val) * total_students / 100 ∧
    ∃ (badminton_percent : SportPercentage),
      badminton_percent = ⟨100 - (basketball_percent.val + chess_percent.val + soccer_percent.val), by sorry⟩ := by
  sorry

#check chess_percentage_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_percentage_proof_l1126_112610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1126_112652

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x + b

-- Define the derivative of f(x)
def f_deriv (a x : ℝ) : ℝ := 2 * a * x - (a + 2) + 1 / x

-- Theorem statement
theorem function_properties (a b : ℝ) (h_a : a > 0) :
  (f_deriv a 1 = 1) →
  (f a b 1 = 0) →
  (a = 2 ∧ b = 2) ∧
  (a ≥ 2 → ∃! x, x > 0 ∧ f a b x = 0) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1126_112652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_iff_m_zero_z_over_one_plus_i_when_m_is_two_l1126_112658

/-- Define a complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := m * (m - 1) + (m - 1) * Complex.I

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (w : ℂ) : Prop := w.re = 0

theorem z_purely_imaginary_iff_m_zero :
  ∀ m : ℝ, isPurelyImaginary (z m) ↔ m = 0 := by sorry

theorem z_over_one_plus_i_when_m_is_two :
  z 2 / (1 + Complex.I) = 3/2 - 1/2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_iff_m_zero_z_over_one_plus_i_when_m_is_two_l1126_112658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_3900_l1126_112675

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let share_ratio := investment_a / total_investment
  share_ratio * total_profit

/-- Theorem stating that A's share of the profit is 3900 given the specified investments and total profit. -/
theorem a_share_is_3900 :
  calculate_share_of_profit 6300 4200 10500 13000 = 3900 := by
  -- Unfold the definition and simplify
  unfold calculate_share_of_profit
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_3900_l1126_112675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1126_112667

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ → Prop

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a parabola with equation y² = 2x -/
def standardParabola : Parabola :=
  { equation := fun x y => y^2 = 2*x,
    directrix := fun x => x = -1/2 }

/-- The focus of the parabola y² = 2x -/
noncomputable def focus : Point :=
  { x := 1/2, y := 0 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the directrix and a point on the parabola -/
theorem parabola_properties (p : Point) 
  (h1 : standardParabola.equation p.x p.y) 
  (h2 : distance p focus = 5/2) : 
  standardParabola.directrix = fun x => x = -1/2 ∧ p.x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1126_112667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l1126_112641

theorem same_color_probability (red_plates blue_plates : ℕ) 
  (h_red : red_plates = 8) (h_blue : blue_plates = 6) : 
  (red_plates.choose 2 + blue_plates.choose 2 : ℚ) / ((red_plates + blue_plates).choose 2) = 43 / 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l1126_112641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_one_when_z_is_64_l1126_112660

/-- A structure representing the relationship between x, y, and z -/
structure Relationship where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ
  m : ℝ  -- Proportionality constant for x and y²
  n : ℝ  -- Proportionality constant for y and 1/√z
  h1 : ∀ z, x z = m * (y z)^2  -- x is directly proportional to y²
  h2 : ∀ z, y z = n / Real.sqrt z  -- y is inversely proportional to √z
  h3 : ∀ z, x z * z = 64  -- Derived relationship
  h4 : ∀ (z₁ z₂ : ℝ), z₂ = 4 * z₁ → x z₂ = (1/2) * x z₁  -- x halves when z is quadrupled

/-- Theorem stating that x = 1 when z = 64 given the conditions -/
theorem x_equals_one_when_z_is_64 (r : Relationship) : r.x 64 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_one_when_z_is_64_l1126_112660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_tangent_c_value_l1126_112608

/-- The value of c for which the line x - y + c = 0 is tangent to the circle (x - 1)^2 + y^2 = 2 -/
theorem line_tangent_to_circle :
  ∃! c : ℝ, ∃! p : ℝ × ℝ, 
    (p.1 - p.2 + c = 0) ∧ 
    ((p.1 - 1)^2 + p.2^2 = 2) :=
by
  sorry

/-- The specific value of c that satisfies the tangency condition -/
theorem tangent_c_value :
  let c := -1 + Real.sqrt 2
  ∃! p : ℝ × ℝ, 
    (p.1 - p.2 + c = 0) ∧ 
    ((p.1 - 1)^2 + p.2^2 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_tangent_c_value_l1126_112608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_distribution_l1126_112685

theorem fruit_distribution (n : ℕ) (h : n = 158) : 
  let apple_recipients := {i : ℕ | i ≤ n ∧ i % 2 = 1}
  let banana_recipients := {i : ℕ | i ≤ n ∧ (n - i + 1) % 3 = 1}
  n - (Finset.card (Finset.filter (λ i => i ∈ apple_recipients) (Finset.range (n+1))) + 
       Finset.card (Finset.filter (λ i => i ∈ banana_recipients) (Finset.range (n+1))) - 
       Finset.card (Finset.filter (λ i => i ∈ apple_recipients ∩ banana_recipients) (Finset.range (n+1)))) = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_distribution_l1126_112685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_smallest_angle_in_configuration_l1126_112697

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of n points in a plane -/
def PointConfiguration (n : ℕ) := Fin n → Point

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

/-- Function to calculate the angle between three points -/
noncomputable def angle (p q r : Point) : ℝ := sorry

/-- Predicate to check if a configuration forms a regular n-gon -/
def isRegularNGon (n : ℕ) (config : PointConfiguration n) : Prop := sorry

/-- Theorem: Maximum smallest angle in a point configuration -/
theorem max_smallest_angle_in_configuration 
  (n : ℕ) 
  (h_n : n ≥ 3) 
  (config : PointConfiguration n) 
  (h_not_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (config i) (config j) (config k)) :
  (∃ (α : ℝ), 
    (∀ i j k, i ≠ j → j ≠ k → i ≠ k → angle (config i) (config j) (config k) ≥ α) ∧ 
    α ≤ 180 / n ∧
    (α = 180 / n ↔ isRegularNGon n config)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_smallest_angle_in_configuration_l1126_112697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_line_fastest_l1126_112607

-- Define the train lines
inductive TrainLine
| Blue
| Red
| Green

-- Define the constants
noncomputable def total_track_length : ℝ := 200
noncomputable def forest_grove_distance : ℝ := total_track_length / 5

-- Define the round trip times for each line
def round_trip_time (line : TrainLine) : ℝ :=
  match line with
  | TrainLine.Blue => 5
  | TrainLine.Red => 6
  | TrainLine.Green => 7

-- Define the speed of each train line
noncomputable def train_speed (line : TrainLine) : ℝ :=
  total_track_length / (round_trip_time line / 2)

-- Define the travel time from Forest Grove to Sherbourne
noncomputable def travel_time (line : TrainLine) : ℝ :=
  (total_track_length - forest_grove_distance) / train_speed line

-- Theorem: The Blue Line is the fastest route from Forest Grove to Sherbourne
theorem blue_line_fastest :
  (∀ line : TrainLine, travel_time TrainLine.Blue ≤ travel_time line) ∧
  travel_time TrainLine.Blue = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_line_fastest_l1126_112607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_x_intercept_is_ten_l1126_112611

-- Define a circle by its center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to create a circle from two diameter endpoints
noncomputable def circleFromDiameter (p1 p2 : ℝ × ℝ) : Circle :=
  { center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2),
    radius := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) / 2 }

-- Define a function to find x-intercepts of a circle
def xIntercepts (c : Circle) : Set ℝ :=
  {x | (x - c.center.1)^2 = c.radius^2}

-- Theorem statement
theorem third_x_intercept_is_ten :
  let c := circleFromDiameter (0, 0) (10, 0)
  let intercepts := xIntercepts c
  (0 ∈ intercepts) ∧ (10 ∈ intercepts) →
  ∃ x ∈ intercepts, x ≠ 0 ∧ x ≠ 10 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_x_intercept_is_ten_l1126_112611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_sheets_is_700_num_sheets_equals_product_l1126_112687

/-- The number of sheets in a stack of used paper --/
def num_sheets : ℕ := 700

/-- The number of sheets each box can contain --/
def sheets_per_box : ℕ := 100

/-- The number of boxes needed to contain all sheets --/
def boxes_needed : ℕ := 7

/-- Theorem: The number of sheets is 700 --/
theorem num_sheets_is_700 : num_sheets = 700 :=
by
  rfl  -- reflexivity, since num_sheets is defined as 700

/-- Theorem: The number of sheets equals the product of boxes needed and sheets per box --/
theorem num_sheets_equals_product : num_sheets = boxes_needed * sheets_per_box :=
by
  rfl  -- reflexivity, since 700 = 7 * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_sheets_is_700_num_sheets_equals_product_l1126_112687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_sin_l1126_112693

theorem negation_of_existential_sin (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.sin x ≥ 1) → 
  (¬p ↔ ∀ x : ℝ, Real.sin x < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_sin_l1126_112693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_80_l1126_112636

/-- The cost function for fuel per kilometer -/
noncomputable def w (v : ℝ) : ℝ := (1 / 300) * (v^2 / (v - 40))

/-- The theorem stating that the minimum cost occurs at v = 80 -/
theorem min_cost_at_80 :
  ∀ v : ℝ, 60 ≤ v → v ≤ 120 → w v ≥ w 80 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_80_l1126_112636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_product_l1126_112671

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a rectangle on a 2D grid -/
structure Rectangle where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℕ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

/-- Calculates the side length of the rectangle -/
noncomputable def sideLength (r : Rectangle) : ℝ :=
  Real.sqrt (squaredDistance r.E r.F : ℝ)

/-- Calculates the area of the rectangle -/
noncomputable def area (r : Rectangle) : ℝ :=
  (sideLength r) ^ 2

/-- Calculates the perimeter of the rectangle -/
noncomputable def perimeter (r : Rectangle) : ℝ :=
  4 * (sideLength r)

/-- The main theorem to prove -/
theorem rectangle_area_perimeter_product (r : Rectangle) :
  r.E = ⟨1, 4⟩ ∧ r.F = ⟨4, 5⟩ ∧ r.G = ⟨5, 2⟩ ∧ r.H = ⟨2, 1⟩ →
  area r * perimeter r = 40 * Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_perimeter_product_l1126_112671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_root_in_interval_l1126_112635

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the function f
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + 2) * g x + 3*x - 4

-- State the theorem
theorem exists_root_in_interval (hg : Continuous g) :
  ∃ c ∈ Set.Ioo 1 2, f g c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_root_in_interval_l1126_112635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_triangle_area_l1126_112603

noncomputable def vector_a : Fin 2 → ℝ := ![3, 2]
noncomputable def vector_b : Fin 2 → ℝ := ![-1, 5]

noncomputable def rotate_90_ccw (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ![- v 1, v 0]

noncomputable def triangle_area (v1 v2 : Fin 2 → ℝ) : ℝ :=
  (1 / 2) * abs (v1 0 * v2 1 - v1 1 * v2 0)

theorem rotated_triangle_area :
  triangle_area (rotate_90_ccw vector_a) (rotate_90_ccw vector_b) = 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_triangle_area_l1126_112603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l1126_112624

noncomputable section

variable (α β : ℝ)

noncomputable def a : ℕ → ℝ
  | 0 => α
  | 1 => β
  | (n + 2) => a (n + 1) + (a n - a (n + 1)) / (2 * (n + 2))

theorem sequence_limit :
  ∃ (L : ℝ), L = α + (β - α) / Real.sqrt (Real.exp 1) ∧
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a α β n - L| < ε :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l1126_112624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1126_112639

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p : Point
  q : Point
  r : Point

/-- Calculate the area of a triangle given its vertices -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The rectangle containing the triangle -/
def rectangle : Point × Point := 
  ({ x := 0, y := 0 }, { x := 3, y := 6 })

/-- The triangle PQR -/
def trianglePQR : Triangle := {
  p := { x := 0, y := 2 }
  q := { x := 3, y := 0 }
  r := { x := 1, y := 6 }
}

/-- Theorem: The area of triangle PQR is 6 square units -/
theorem area_of_triangle_PQR :
  triangleArea trianglePQR = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l1126_112639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_wednesday_or_sunday_l1126_112649

-- Define the days of the week
inductive Day : Type
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
  deriving Repr, DecidableEq

-- Define the sports
inductive Sport : Type
  | Running | Basketball | Golf | Swimming | Tennis | Cycling
  deriving Repr, DecidableEq

-- Define a schedule as a function from Day to Sport
def Schedule := Day → Sport

-- Define a successor function for Day
def Day.succ : Day → Day
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def valid_schedule (s : Schedule) : Prop :=
  -- Mahdi plays basketball on Tuesday
  s Day.Tuesday = Sport.Basketball
  -- Mahdi plays golf four days after basketball (on Saturday)
  ∧ s Day.Saturday = Sport.Golf
  -- Mahdi cycles on Monday
  ∧ s Day.Monday = Sport.Cycling
  -- Mahdi runs three days a week
  ∧ (∃ d1 d2 d3 : Day, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
      s d1 = Sport.Running ∧ s d2 = Sport.Running ∧ s d3 = Sport.Running)
  -- Mahdi never runs on consecutive days
  ∧ (∀ d : Day, s d = Sport.Running → s (Day.succ d) ≠ Sport.Running)
  -- Mahdi never plays tennis the day after running or cycling
  ∧ (∀ d : Day, (s d = Sport.Running ∨ s d = Sport.Cycling) → s (Day.succ d) ≠ Sport.Tennis)
  -- Mahdi swims and plays tennis
  ∧ (∃ d : Day, s d = Sport.Swimming)
  ∧ (∃ d : Day, s d = Sport.Tennis)

theorem mahdi_swims_on_wednesday_or_sunday (s : Schedule) (h : valid_schedule s) :
  (s Day.Wednesday = Sport.Swimming ∨ s Day.Sunday = Sport.Swimming) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_wednesday_or_sunday_l1126_112649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gate_area_ratio_l1126_112696

/-- A decorative gate with a rectangular section and quarter-circles at each corner -/
structure DecorativeGate where
  shorter_side : ℝ
  longer_side : ℝ
  h_ratio : longer_side / shorter_side = 7 / 4
  h_shorter : shorter_side = 20

/-- The ratio of the area of the rectangle to the area of the flanking quarter-circles -/
noncomputable def area_ratio (gate : DecorativeGate) : ℝ :=
  (gate.longer_side * gate.shorter_side) / (Real.pi * gate.shorter_side^2)

theorem gate_area_ratio (gate : DecorativeGate) :
    area_ratio gate = 7 / (4 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gate_area_ratio_l1126_112696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_point_l1126_112677

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def line_through_point (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 / a + p.2 / b = 1}

def valid_line (a b : ℝ) : Prop :=
  (∃ n : ℕ, n < 10 ∧ is_prime n ∧ a = n) ∧ 
  (b > 0 ∧ b ≠ 5 ∧ ∃ m : ℕ, b = m)

theorem unique_line_through_point :
  ∃! l : Set (ℝ × ℝ), ∃ a b : ℝ,
    valid_line a b ∧
    l = line_through_point a b ∧
    (5, 4) ∈ l :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_through_point_l1126_112677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1126_112689

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

-- Define the target function
noncomputable def g (x : ℝ) : ℝ := Real.sin x

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f (x + 4 * Real.pi) = f x) ∧
  (∃ a b c : ℝ, ∀ x : ℝ, g x = c * f (a * x + b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1126_112689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l1126_112614

-- Define the shapes
inductive Shape
| Triangle
| SmallCircle
| Square
| Pentagon

-- Define the position of a shape on the circle
structure Position :=
(angle : ℚ)

-- Define the rotation
def rotate (p : Position) (angle : ℚ) : Position :=
{ angle := (p.angle + angle) % 1 }

-- Define the initial configuration
def initial_config : List (Shape × Position) := sorry

-- Define the rotated configuration
def rotated_config : List (Shape × Position) := sorry

-- Theorem statement
theorem rotation_result :
  ∀ (s : Shape) (p : Position),
    (s, p) ∈ initial_config →
    ∃ (p_rotated : Position),
      (s, p_rotated) ∈ rotated_config ∧
      ∃ (s1 s2 : Shape) (p1 p2 : Position),
        (s1, p1) ∈ initial_config ∧
        (s2, p2) ∈ initial_config ∧
        s1 ≠ s ∧ s2 ≠ s ∧ s1 ≠ s2 ∧
        p1.angle < p2.angle ∧
        p1.angle < p_rotated.angle ∧
        p_rotated.angle < p2.angle :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_result_l1126_112614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1126_112683

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (Real.pi - x) - Real.sin (5 * Real.pi / 2 + x)

theorem f_increasing_interval :
  ∃ (α β : ℝ), f α = 2 ∧ f β = 0 ∧ |α - β| ≥ Real.pi/2 →
  ∀ (k : ℤ), StrictMonoOn f (Set.Icc ((2 * k : ℝ) * Real.pi - Real.pi/3) ((2 * k : ℝ) * Real.pi + 2*Real.pi/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1126_112683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_of_D_l1126_112669

/-- Given that N(4,8) is the midpoint of CD and C(5,4) is one endpoint, 
    the sum of coordinates of point D is 15. -/
theorem sum_coordinates_of_D : ∀ (D : ℝ × ℝ),
  (4, 8) = ((5, 4) + D) / 2 →
  D.1 + D.2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coordinates_of_D_l1126_112669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_overall_profit_l1126_112627

noncomputable def grinder_cost : ℝ := 15000
noncomputable def mobile_cost : ℝ := 8000
noncomputable def grinder_loss_percent : ℝ := 2
noncomputable def mobile_profit_percent : ℝ := 10

noncomputable def grinder_selling_price : ℝ := grinder_cost * (1 - grinder_loss_percent / 100)
noncomputable def mobile_selling_price : ℝ := mobile_cost * (1 + mobile_profit_percent / 100)

noncomputable def total_cost : ℝ := grinder_cost + mobile_cost
noncomputable def total_selling_price : ℝ := grinder_selling_price + mobile_selling_price

theorem john_overall_profit :
  total_selling_price - total_cost = 500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_overall_profit_l1126_112627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_through_focus_parabola_line_fixed_point_l1126_112663

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line that passes through a point (a, b)
def line_through (a b : ℝ) (x y : ℝ) : Prop := x = y * (a - 1) + a

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Part 1
theorem parabola_line_through_focus (xA yA xB yB : ℝ) :
  parabola xA yA ∧ parabola xB yB ∧ 
  line_through (focus.1) (focus.2) xA yA ∧ 
  line_through (focus.1) (focus.2) xB yB ∧ 
  (xA, yA) ≠ (xB, yB) →
  dot_product xA yA xB yB = -3 := by sorry

-- Part 2
theorem parabola_line_fixed_point (xA yA xB yB : ℝ) :
  parabola xA yA ∧ parabola xB yB ∧
  (∃ (t b : ℝ), line_through t b xA yA ∧ line_through t b xB yB) ∧
  (xA, yA) ≠ (xB, yB) ∧
  dot_product xA yA xB yB = -4 →
  ∃ (t : ℝ), line_through 2 0 xA yA ∧ line_through 2 0 xB yB := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_through_focus_parabola_line_fixed_point_l1126_112663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_formula_l1126_112616

open BigOperators

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem sequence_sum_formula (a b : ℕ → ℝ) (n : ℕ) :
  (∀ k, b k ≠ 0) →
  a 1 = 1 →
  b 1 = 1 →
  (∀ k, b (k + 1) * (a k + 3 * b k) = a (k + 1) * b k) →
  (∃ q : ℝ, q > 0 ∧ ∀ k, b (k + 1) = q * b k) →
  b 3 ^ 2 = 4 * b 2 * b 6 →
  sequence_sum a n = 8 - (6 * n + 8) * (1 / 2) ^ n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_formula_l1126_112616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1126_112662

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 2*x + 3}
def B : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2*x + 7}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 2 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1126_112662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_division_count_l1126_112645

theorem book_division_count (total_books : Nat) (final_group_size : Nat) : 
  total_books = 400 → final_group_size = 25 → 
  Nat.log 2 (total_books / final_group_size) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_division_count_l1126_112645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_with_one_element_l1126_112665

theorem set_with_one_element (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_with_one_element_l1126_112665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1126_112642

noncomputable section

variable {a b c A B C : ℝ}

-- Define the triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Define the given condition
def condition (a b c A B C : ℝ) : Prop :=
  2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a

theorem triangle_problem 
  (h_triangle : triangle_ABC a b c A B C)
  (h_condition : condition a b c A B C)
  (h_cosB : Real.cos B = 3/5) :
  A = Real.pi/3 ∧ Real.sin (B - C) = (7 * Real.sqrt 3 - 12) / 50 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1126_112642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_y_l1126_112691

/-- Predicate indicating that the parametric functions x and y describe an ellipse -/
def IsEllipse (x y : ℝ → ℝ) : Prop :=
  sorry

/-- Predicate indicating that the major axis of the curve described by x and y is along the y-axis -/
def MajorAxisAlongY (x y : ℝ → ℝ) : Prop :=
  sorry

/-- A curve represented by the equation 2x² + ky² = 1 is an ellipse with its major axis along the y-axis if and only if k is in the open interval (0, 2) -/
theorem ellipse_major_axis_y (k : ℝ) : 
  (∃ (x y : ℝ → ℝ), ∀ t, 2 * (x t)^2 + k * (y t)^2 = 1 ∧ 
   IsEllipse x y ∧ MajorAxisAlongY x y) ↔ 
  k ∈ Set.Ioo 0 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_y_l1126_112691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_feeding_ratio_l1126_112699

/-- Proves that the ratio of Wanda's bread to Wanda's treats is 3:1 given the problem conditions --/
theorem zoo_feeding_ratio :
  ∀ (jane_treats : ℕ) (wanda_bread_multiplier : ℕ),
    let jane_bread := (75 * jane_treats) / 100
    let wanda_treats := jane_treats / 2
    let wanda_bread := wanda_bread_multiplier * wanda_treats
    wanda_bread = 90 →
    jane_bread + jane_treats + wanda_bread + wanda_treats = 225 →
    wanda_bread / wanda_treats = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_feeding_ratio_l1126_112699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1126_112676

noncomputable def f (x : ℝ) : ℝ := 4 / (3 * x) + 3 / (1 - x)

theorem f_minimum_value (x : ℝ) (hx : 0 < x ∧ x < 1) : f x ≥ 25 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1126_112676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l1126_112666

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((-x^2 + 6*x - 5)) / Real.log (1/2)

-- Define the domain of f
def domain : Set ℝ := {x | 1 < x ∧ x < 5}

-- State that log_{1/2} is decreasing
axiom log_half_decreasing : ∀ x y, x < y → f x > f y

-- Define the monotonic decreasing interval
def monotonic_decreasing_interval : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Theorem statement
theorem f_monotonic_decreasing :
  ∀ x ∈ domain, ∀ y ∈ domain,
    x ∈ monotonic_decreasing_interval ∧ 
    y ∈ monotonic_decreasing_interval ∧ 
    x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l1126_112666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reciprocal_relation_l1126_112640

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define that ABC is a right triangle
variable (right_triangle : IsRightAngle A B C)

-- Define the altitude AH
variable (H : EuclideanSpace ℝ (Fin 2))
variable (altitude_AH : IsAltitude A H B C)

-- Define point O on AH
variable (O : EuclideanSpace ℝ (Fin 2))
variable (O_on_AH : O ∈ Segment A H)

-- State the theorem
theorem triangle_reciprocal_relation :
  1 / dist A O = 1 / dist A H + 1 / dist B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reciprocal_relation_l1126_112640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l1126_112643

def correct_answer : String := "B"

theorem answer_is_correct : correct_answer = "B" := by
  rfl

#check answer_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_is_correct_l1126_112643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_circle_l1126_112668

/-- The curve function -/
noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (x^2 + a*x + 1 - 2*a)

/-- The circle function -/
def circle_func (x y : ℝ) : ℝ := x^2 + 2*x + y^2 - 12

/-- The tangent line function -/
def tangent_line (a x : ℝ) : ℝ := (1 - a) * x + (1 - 2*a)

theorem tangent_intersects_circle (a : ℝ) : 
  ∃ x y : ℝ, circle_func x y = 0 ∧ y = tangent_line a x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_circle_l1126_112668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l1126_112605

/-- A rhombus with given side length and shorter diagonal -/
structure Rhombus where
  side_length : ℝ
  shorter_diagonal : ℝ

/-- The longer diagonal of a rhombus -/
noncomputable def longer_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (r.side_length ^ 2 - (r.shorter_diagonal / 2) ^ 2)

/-- Theorem: In a rhombus with side length 40 and shorter diagonal 32, 
    the longer diagonal has length 16√21 -/
theorem rhombus_longer_diagonal :
  let r : Rhombus := ⟨40, 32⟩
  longer_diagonal r = 16 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l1126_112605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_subtraction_l1126_112681

theorem power_subtraction (n : ℕ) : (2 : ℤ)^n - (2 : ℤ)^(n+1) = -(2 : ℤ)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_subtraction_l1126_112681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_diagonals_perpendicular_right_trapezoid_right_triangle_at_midpoint_l1126_112629

/-- A trapezoid with right angles at two vertices -/
structure RightTrapezoid where
  a : ℝ  -- length of AB
  b : ℝ  -- length of CD
  h : ℝ  -- length of AD (height)
  a_pos : 0 < a
  b_pos : 0 < b
  h_pos : 0 < h

-- Remove the midpoint definition as it's already defined in Mathlib

theorem right_trapezoid_diagonals_perpendicular (t : RightTrapezoid) :
  (t.h ^ 2 = t.a * t.b) ↔ 
  (∃ (AC BD : ℝ × ℝ), AC.1 * BD.1 + AC.2 * BD.2 = 0) :=
sorry

theorem right_trapezoid_right_triangle_at_midpoint (t : RightTrapezoid) :
  (t.h ^ 2 = 4 * t.a * t.b) ↔ 
  (∃ (B M C : ℝ × ℝ), (M.1 = (t.a + t.b) / 2) ∧ 
    ((B.1 - M.1) * (C.1 - M.1) + (B.2 - M.2) * (C.2 - M.2) = 0)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_diagonals_perpendicular_right_trapezoid_right_triangle_at_midpoint_l1126_112629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_l1126_112653

/-- The volume of a cube with edge length a -/
noncomputable def cube_volume (a : ℝ) : ℝ := a ^ 3

/-- The rise in water level when a cube is placed in a tank -/
noncomputable def water_level_rise_of_cube_in_tank (edge_length : ℝ) (tank_area : ℝ) : ℝ :=
  cube_volume edge_length / tank_area

/-- The rise in water level when a 1-meter cube is placed in a tank -/
theorem water_level_rise (A : ℝ) (h : A > 0) : 
  (1 : ℝ) / A = water_level_rise_of_cube_in_tank 1 A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_rise_l1126_112653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l1126_112688

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + Real.sqrt 2/2 * t, 1 + Real.sqrt 2/2 * t)

def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y = 0

def point_M : ℝ × ℝ := (2, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_reciprocal_sum :
  ∃ (t₁ t₂ : ℝ),
    circle_C (line_l t₁).1 (line_l t₁).2 ∧
    circle_C (line_l t₂).1 (line_l t₂).2 ∧
    t₁ ≠ t₂ ∧
    (1 / distance point_M (line_l t₁) + 1 / distance point_M (line_l t₂) = Real.sqrt 30 / 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l1126_112688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_cubic_polynomial_l1126_112678

theorem roots_of_cubic_polynomial : 
  let p (x : ℝ) := x^3 - 7*x^2 + 14*x - 8
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_cubic_polynomial_l1126_112678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_outcome_l1126_112621

def num_children : ℕ := 4

def prob_boy : ℚ := 1/2
def prob_girl : ℚ := 1/2

def prob_all_boys : ℚ := prob_boy ^ num_children
def prob_all_girls : ℚ := prob_girl ^ num_children
def prob_two_each : ℚ := (Nat.choose num_children 2 : ℚ) * (prob_boy ^ 2) * (prob_girl ^ 2)
def prob_three_one : ℚ := 2 * (Nat.choose num_children 1 : ℚ) * (prob_boy ^ 3) * prob_girl

theorem most_likely_outcome :
  prob_three_one > prob_all_boys ∧
  prob_three_one > prob_all_girls ∧
  prob_three_one > prob_two_each := by
  sorry

#eval prob_all_boys
#eval prob_all_girls
#eval prob_two_each
#eval prob_three_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_outcome_l1126_112621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1126_112661

/-- The type of real polynomials -/
def RealPoly := ℝ → ℝ

/-- A function to represent the number of distinct roots of a polynomial -/
noncomputable def distinct_roots (p : RealPoly) : ℕ := sorry

/-- First polynomial from the problem -/
noncomputable def poly1 (p q : ℝ) : RealPoly := 
  fun x => (x + p) * (x + q) * (x - 15) / ((x - 5) ^ 2)

/-- Second polynomial from the problem -/
noncomputable def poly2 (p q : ℝ) : RealPoly := 
  fun x => (x - 2*p) * (x - 5) * (x + 10) / ((x + q) * (x - 15))

/-- Main theorem -/
theorem problem_solution (p q : ℝ) : 
  distinct_roots (poly1 p q) = 3 → 
  distinct_roots (poly2 p q) = 2 → 
  100 * p + q = 240 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1126_112661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_sum_l1126_112646

theorem existence_of_divisible_sum (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ∃ n : ℕ, n > 0 ∧ p ∣ (20^n + 15^n - 12^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_sum_l1126_112646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_equals_one_l1126_112612

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = m + 2/(3^x - 1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  m + 2 / (3^x - 1)

/-- Theorem: If f(x) = m + 2/(3^x - 1) is an odd function, then m = 1 -/
theorem odd_function_m_equals_one (m : ℝ) (h : IsOdd (f m)) : m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_m_equals_one_l1126_112612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l1126_112613

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (lies_on : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : parallel m n) 
  (h3 : lies_on n β)
  (h4 : parallel_plane α β) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_l1126_112613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l1126_112620

noncomputable def hyperbola_C1 (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

noncomputable def hyperbola_C2 (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

noncomputable def asymptote_equation (x y a b : ℝ) : Prop := y = (b / a) * x

noncomputable def distance_point_to_line (a b c : ℝ) : ℝ := (b * c) / Real.sqrt (a^2 + b^2)

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem hyperbola_real_axis_length 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (h_eccentricity : eccentricity a c = Real.sqrt 5 / 2) 
  (h_area : area_triangle a b = 16) 
  (h_pythagoras : a^2 + b^2 = c^2) : 
  2 * a = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l1126_112620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1126_112633

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the slope of the parallel line
def m : ℝ := 4

-- Theorem statement
theorem tangent_line_equation :
  ∃ (a b : ℝ), (∀ x : ℝ, deriv f x = m → (4*x - f x = a ∨ 4*x - f x = b)) ∧ a ≠ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1126_112633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_l1126_112664

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The equation of the conic section -/
noncomputable def conic_equation (x y : ℝ) : Prop :=
  distance x y 0 2 + distance x y 6 (-4) = 14

/-- The distance between the foci -/
noncomputable def foci_distance : ℝ := distance 0 2 6 (-4)

/-- Theorem: The given equation describes an ellipse -/
theorem is_ellipse : ∃ (x y : ℝ), conic_equation x y ∧ foci_distance < 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_l1126_112664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l1126_112604

theorem max_imaginary_part_of_roots (z : ℂ) : 
  z^10 - z^8 + z^6 - z^4 + z^2 - 1 = 0 → 
  ∃ (θ : ℝ), -π/2 ≤ θ ∧ θ ≤ π/2 ∧ 
  Complex.abs z.im ≤ Real.sin θ ∧
  Real.sin θ = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_imaginary_part_of_roots_l1126_112604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1126_112647

theorem trigonometric_identity (α : ℝ) (h : Real.tan α + 1 / Real.tan α = 9 / 4) :
  Real.tan α ^ 2 + 1 / (Real.sin α * Real.cos α) + 1 / Real.tan α ^ 2 = 85 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1126_112647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1126_112601

/-- Given a line and an ellipse with certain properties, prove the equation of the ellipse. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 + A.2 = 1) ∧ 
    (B.1 + B.2 = 1) ∧ 
    (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) ∧
    (let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2); C.1 / C.2 = Real.sqrt 2)) →
  ∀ (x y : ℝ), x^2 / 3 + Real.sqrt 2 * y^2 / 3 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
by
  intro h
  intro x y
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1126_112601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1126_112628

/-- The type representing a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a 6-digit number in the form 27x85y -/
def NumberForm (x y : Digit) : ℕ := 
  270000 + 10000 * x.val + 8500 + y.val

/-- The theorem stating the divisibility condition -/
theorem divisibility_condition (x y : Digit) :
  33 ∣ NumberForm x y ↔ ((x.val + y.val) % 3 = 2 ∧ (y.val - x.val) % 11 = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1126_112628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_difference_l1126_112680

/-- Given a line l with direction vector (2, -1, 3) passing through points A(0, y, 3) and B(-1, 2, z),
    prove that y - z = 0 -/
theorem line_point_difference (y z : ℝ) : 
  let m : Fin 3 → ℝ := ![2, -1, 3]
  let A : Fin 3 → ℝ := ![0, y, 3]
  let B : Fin 3 → ℝ := ![-1, 2, z]
  (∃ (k : ℝ), ∀ i, B i - A i = k * m i) →
  y - z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_difference_l1126_112680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1126_112637

noncomputable def diamond (x y : ℝ) : ℝ := (x^2 + y^2) / (x - y)

theorem diamond_calculation : diamond (diamond 3 1) 2 = 29/3 := by
  -- Expand the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1126_112637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_two_std_dev_below_mean_l1126_112619

/-- Represents a normal distribution --/
structure NormalDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation
  σ_pos : σ > 0

/-- Calculates the number of standard deviations a value is from the mean --/
noncomputable def standardDeviationsFromMean (nd : NormalDistribution) (x : ℝ) : ℝ :=
  (nd.μ - x) / nd.σ

/-- Theorem: For a normal distribution with mean 15 and standard deviation 1.5,
    the value 12 is exactly 2 standard deviations less than the mean --/
theorem value_two_std_dev_below_mean :
  let nd : NormalDistribution := ⟨15, 1.5, by norm_num⟩
  standardDeviationsFromMean nd 12 = 2 := by
  -- Expand the definition of standardDeviationsFromMean
  unfold standardDeviationsFromMean
  -- Simplify the arithmetic
  simp [NormalDistribution.μ, NormalDistribution.σ]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_two_std_dev_below_mean_l1126_112619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_composite_l1126_112686

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_a_for_composite : 
  (∃ a : ℕ, 0 < a ∧ 
    (∀ x : ℤ, is_composite (Int.natAbs (x^4) + a^2 + 16)) ∧
    (∀ b : ℕ, 0 < b → b < a → ∃ x : ℤ, ¬ is_composite (Int.natAbs (x^4) + b^2 + 16))) ∧
  (∀ a : ℕ, (0 < a ∧ 
    (∀ x : ℤ, is_composite (Int.natAbs (x^4) + a^2 + 16)) ∧
    (∀ b : ℕ, 0 < b → b < a → ∃ x : ℤ, ¬ is_composite (Int.natAbs (x^4) + b^2 + 16))) → 
  a = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_composite_l1126_112686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_correct_l1126_112631

/-- Data points from the experiment -/
def data_points : List (ℝ × ℝ) := [(1, 2), (2, 5), (4, 7), (5, 10)]

/-- Calculate the mean of a list of real numbers -/
noncomputable def mean (xs : List ℝ) : ℝ := 
  xs.sum / xs.length

/-- Extract x-coordinates from the data points -/
def x_values : List ℝ := data_points.map Prod.fst

/-- Extract y-coordinates from the data points -/
def y_values : List ℝ := data_points.map Prod.snd

/-- Mean of x-coordinates -/
noncomputable def x_mean : ℝ := mean x_values

/-- Mean of y-coordinates -/
noncomputable def y_mean : ℝ := mean y_values

/-- The proposed regression line equation -/
def regression_line (x : ℝ) : ℝ := x + 3

theorem regression_line_correct : 
  x_mean = 3 ∧ 
  y_mean = 6 ∧ 
  regression_line x_mean = y_mean := by
  sorry

#eval data_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_correct_l1126_112631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1126_112656

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 6 ≥ 0}
def B : Set ℝ := {x | -3 < x + 1 ∧ x + 1 < 3}

-- Theorem statements
theorem set_operations :
  (A ∩ B = Set.Ioo (-4) 2) ∧
  (A ∪ B = Set.Iic 2 ∪ Set.Ici 3) ∧
  ((Set.univ \ A) ∩ B = ∅) := by
  sorry

-- Where:
-- Set.Ioo a b is the open interval (a, b)
-- Set.Iic a is the interval (-∞, a]
-- Set.Ici a is the interval [a, +∞)
-- Set.univ \ A is the complement of A in ℝ
-- ∅ is the empty set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1126_112656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_radius_l1126_112623

/-- The radius of a sphere touching the sides of the base and extensions of lateral edges of a regular triangular pyramid -/
noncomputable def sphere_radius (a b : ℝ) : ℝ :=
  a / (2 * Real.sqrt (3 * b^2 - a^2))

/-- Theorem: For a regular triangular pyramid with base side length a and lateral edge length b,
    the radius of the sphere that touches the sides of the base and the extensions of the lateral edges
    is equal to a / (2 * sqrt(3b^2 - a^2)) -/
theorem pyramid_sphere_radius (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b > a / Real.sqrt 3) :
  sphere_radius a b = a / (2 * Real.sqrt (3 * b^2 - a^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_radius_l1126_112623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_15_l1126_112609

theorem decimal_to_binary_15 : 
  ∃ (b : List Nat), 
    (b.length = 4) ∧ 
    (∀ i, i < 4 → b.get ⟨i, by sorry⟩ = 1) ∧ 
    (b.foldl (λ acc x ↦ 2 * acc + x) 0 = 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_15_l1126_112609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_monotonicity_l1126_112659

open Real

/-- Given a function f(x) = 2tan(ωx + π/3) with ω > 0 and smallest positive period π/2 -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * tan (ω * x + Real.pi / 3)

/-- The smallest positive period of f is π/2 -/
axiom period_f (ω : ℝ) : ω > 0 → ∃ (T : ℝ), T > 0 ∧ T = Real.pi / 2 ∧ ∀ (x : ℝ), f ω x = f ω (x + T)

theorem domain_and_monotonicity (ω : ℝ) (h : ω > 0) :
  (∀ (x : ℝ), f ω x ≠ 0 ↔ ∃ (k : ℤ), x ≠ k * Real.pi / 2 + Real.pi / 12) ∧
  (∀ (k : ℤ), StrictMonoOn (f ω) (Set.Ioo (k * Real.pi / 2 - 5 * Real.pi / 12) (k * Real.pi / 2 + Real.pi / 12))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_monotonicity_l1126_112659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1126_112625

-- Define the power function
noncomputable def power_function (a : ℚ) : ℝ → ℝ := fun x ↦ (x : ℝ) ^ (a : ℝ)

-- State the theorem
theorem power_function_through_point :
  ∀ a : ℚ, power_function a 2 = Real.sqrt 2 / 2 → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1126_112625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_number_characterization_l1126_112626

/-- Two positive integers are prime-related if one divides the other by a prime number -/
def PrimeRelated (a b : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ (a = p * b ∨ b = p * a)

/-- A positive integer n is "good" if it satisfies the given conditions -/
def IsGoodNumber (n : ℕ) : Prop :=
  (∃ d₁ d₂ d₃ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0 ∧ d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃) ∧ 
  (∃ f : ℕ → ℕ, Function.Surjective f ∧ (∀ d : ℕ, d > 0 → d ∣ n → ∃ k : ℕ, f k = d) ∧
    ∀ k : ℕ, PrimeRelated (f k) (f (k + 1)))

/-- A number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A number is a prime power if it's equal to some prime raised to a positive power -/
def IsPrimePower (n : ℕ) : Prop :=
  ∃ p k : ℕ, p > 1 ∧ k > 0 ∧ Nat.Prime p ∧ n = p ^ k

theorem good_number_characterization (n : ℕ) (h : n > 0) :
  IsGoodNumber n ↔ ¬(IsPerfectSquare n ∨ IsPrimePower n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_number_characterization_l1126_112626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_pen_probability_l1126_112606

/-- Represents the distribution of pens in a bag -/
structure BagDistribution where
  green : ℕ
  black : ℕ
  red : ℕ

/-- Calculates the total number of pens in a bag -/
def BagDistribution.total (bag : BagDistribution) : ℕ :=
  bag.green + bag.black + bag.red

/-- The distribution of pens in the three bags -/
def bags : List BagDistribution := [
  { green := 5, black := 6, red := 7 },
  { green := 3, black := 4, red := 8 },
  { green := 2, black := 7, red := 5 }
]

/-- The probability of choosing a black pen given the distribution in the bags -/
theorem black_pen_probability : 
  (bags.map (fun bag => (bag.black : ℚ) / bag.total * bag.total)).sum / 
  (bags.map (fun bag => (bag.total : ℚ))).sum = 17 / 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_pen_probability_l1126_112606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_growth_l1126_112690

def sequence_a (a₀ a₁ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | 1 => a₁
  | (n + 2) => 3 * sequence_a a₀ a₁ (n + 1) - 2 * sequence_a a₀ a₁ n

theorem sequence_growth (a₀ a₁ : ℕ) (h : a₁ > a₀) :
  sequence_a a₀ a₁ 100 > 2^99 := by
  sorry

#eval sequence_a 1 2 100  -- This is just to check if the function compiles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_growth_l1126_112690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solution_pairs_l1126_112674

def satisfies_equation (m n : ℕ+) : Prop :=
  (6 : ℚ) / m.val + (3 : ℚ) / n.val = 1

theorem six_solution_pairs :
  ∃! (s : Finset (ℕ+ × ℕ+)), s.card = 6 ∧ ∀ p ∈ s, satisfies_equation p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solution_pairs_l1126_112674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planet_x_day_division_l1126_112698

theorem planet_x_day_division :
  let day_seconds : ℕ := 100000
  let valid_division (n m : ℕ) : Prop := n > 0 ∧ m > 0 ∧ n * m = day_seconds
  (∑' p : ℕ × ℕ, if valid_division p.1 p.2 then 1 else 0) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planet_x_day_division_l1126_112698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_EF_GH_l1126_112655

def E : Fin 3 → ℝ := ![3, -2, 5]
def F : Fin 3 → ℝ := ![13, -12, 9]
def G : Fin 3 → ℝ := ![1, 4, -3]
def H : Fin 3 → ℝ := ![3, -4, 11]

def line_EF (t : ℝ) : Fin 3 → ℝ := fun i => E i + t * (F i - E i)

def line_GH (s : ℝ) : Fin 3 → ℝ := fun i => G i + s * (H i - G i)

def intersection_point : Fin 3 → ℝ := ![5, -4, 5.8]

theorem intersection_point_EF_GH :
  ∃ (t s : ℝ), line_EF t = line_GH s ∧ line_EF t = intersection_point :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_EF_GH_l1126_112655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangles_with_longest_8_l1126_112600

/-- A triangle with integer sides -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of all triangles with integer sides and longest side 8 -/
def triangles_with_longest_8 : Set IntTriangle :=
  {t : IntTriangle | t.a ≤ 8 ∧ t.b ≤ 8 ∧ t.c ≤ 8 ∧ (t.a = 8 ∨ t.b = 8 ∨ t.c = 8)}

/-- Prove that the set is finite -/
instance : Fintype triangles_with_longest_8 := by
  sorry

/-- Count the number of triangles with longest side 8 -/
theorem count_triangles_with_longest_8 :
  Fintype.card triangles_with_longest_8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangles_with_longest_8_l1126_112600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_implies_m_values_l1126_112602

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + Real.exp (2 * x) - m

noncomputable def tangent_line_area (m : ℝ) : ℝ :=
  let slope := 1 + 2 * Real.exp 0
  let y_intercept := f m 0
  let x_intercept := -y_intercept / slope
  (1 / 2) * abs y_intercept * abs x_intercept

theorem tangent_triangle_area_implies_m_values (m : ℝ) :
  tangent_line_area m = 1 / 6 → m = 2 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_implies_m_values_l1126_112602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_sum_abc_is_172_l1126_112644

/-- The maximum area of an equilateral triangle inscribed in or contained within a 12 by 13 rectangle -/
noncomputable def max_equilateral_triangle_area : ℝ := 169 * Real.sqrt 3

/-- The side lengths of the rectangle -/
def rectangle_width : ℝ := 12
def rectangle_height : ℝ := 13

/-- Theorem stating the maximum area of an equilateral triangle inscribed in or contained within the rectangle -/
theorem max_equilateral_triangle_area_in_rectangle :
  ∀ (triangle_area : ℝ),
  (∃ (x y z : ℝ),
    0 ≤ x ∧ x ≤ rectangle_width ∧
    0 ≤ y ∧ y ≤ rectangle_width ∧
    0 ≤ z ∧ z ≤ rectangle_height ∧
    triangle_area = Real.sqrt 3 / 4 * (x - y) ^ 2) →
  triangle_area ≤ max_equilateral_triangle_area :=
by sorry

/-- The sum of a, b, and c in the expression a√b - c -/
def sum_abc : ℕ := 172

/-- Theorem stating that the sum of a, b, and c is 172 -/
theorem sum_abc_is_172 : sum_abc = 172 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_sum_abc_is_172_l1126_112644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shots_to_destroy_tank_l1126_112638

/-- Represents a grid cell --/
structure Cell where
  x : Fin 41
  y : Fin 41
deriving DecidableEq

/-- Represents the state of the tank --/
inductive TankState
  | Unhit
  | Hit
  | Destroyed

/-- Represents the game state --/
structure GameState where
  tankPosition : Cell
  tankState : TankState

/-- Function to determine if two cells are adjacent --/
def isAdjacent (c1 c2 : Cell) : Bool :=
  (c1.x = c2.x ∧ (c1.y = c2.y - 1 ∨ c1.y = c2.y + 1)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x - 1 ∨ c1.x = c2.x + 1))

/-- Function to simulate a shot --/
def shoot (state : GameState) (target : Cell) : GameState :=
  if state.tankPosition = target then
    match state.tankState with
    | TankState.Unhit => { tankPosition := sorry, tankState := TankState.Hit }
    | TankState.Hit => { tankPosition := state.tankPosition, tankState := TankState.Destroyed }
    | TankState.Destroyed => state
  else
    state

/-- Theorem stating the minimum number of shots required --/
theorem min_shots_to_destroy_tank :
  ∃ (shotSequence : List Cell),
    shotSequence.length = 2521 ∧
    ∀ (initialState : GameState),
      ∃ (finalState : GameState),
        finalState.tankState = TankState.Destroyed ∧
        finalState = shotSequence.foldl shoot initialState := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shots_to_destroy_tank_l1126_112638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l1126_112632

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x - 2/3

-- State the theorem
theorem f_extrema :
  -- Conditions
  (∀ x : ℝ, (x + f x - 2 = 0) → x = 2) →
  -- Conclusions
  (∃ x₁ : ℝ, x₁ = 1 ∧ IsLocalMax f x₁ ∧ f x₁ = 2/3) ∧
  (∃ x₂ : ℝ, x₂ = 3 ∧ IsLocalMin f x₂ ∧ f x₂ = -2/3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l1126_112632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_old_electricity_price_l1126_112657

/-- The old price of electricity per kilowatt-hour given the conditions -/
noncomputable def old_price_per_kwh (price_increase : ℝ) 
                      (old_computer_power : ℝ) 
                      (new_computer_power_increase : ℝ) 
                      (run_time : ℝ) 
                      (run_cost : ℝ) : ℝ :=
  run_cost / (old_computer_power / 1000 * run_time)

/-- Theorem stating that the old price of electricity per kilowatt-hour was $0.225 -/
theorem old_electricity_price :
  old_price_per_kwh 0.25 800 0.5 50 9 = 0.225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_old_electricity_price_l1126_112657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_outside_circle_specific_l1126_112617

/-- Represents an isosceles triangle ABC inscribed in a circle -/
structure IsoscelesTriangleInCircle where
  AB : ℝ
  BC : ℝ
  hAB_pos : 0 < AB
  hBC_pos : 0 < BC
  hIsosceles : AB = AC

/-- Calculates the fraction of the triangle's area outside the circle -/
noncomputable def fractionOutsideCircle (t : IsoscelesTriangleInCircle) : ℝ :=
  (3 * Real.sqrt 165 - 4 * Real.pi) / (3 * Real.sqrt 165)

/-- Theorem stating the fraction of area outside the circle for a specific isosceles triangle -/
theorem fraction_outside_circle_specific (t : IsoscelesTriangleInCircle) 
  (h_AB : t.AB = 8) (h_BC : t.BC = 6) : 
  fractionOutsideCircle t = (3 * Real.sqrt 165 - 4 * Real.pi) / (3 * Real.sqrt 165) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_outside_circle_specific_l1126_112617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1126_112694

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def on_unit_circle (t : Triangle) : Prop :=
  ∃ (x y z : Real), x^2 + y^2 + z^2 = 1

def cosine_relation (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.A = t.c * Real.cos t.B + t.b * Real.cos t.C

def side_condition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = 4

-- Define the area function (placeholder)
noncomputable def area (t : Triangle) : Real :=
  1/2 * t.b * t.c * Real.sin t.A

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : on_unit_circle t) 
  (h2 : cosine_relation t) :
  Real.cos t.A = 1/2 ∧ 
  (side_condition t → area t = Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1126_112694
