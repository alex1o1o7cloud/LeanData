import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l786_78662

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- State the theorem
theorem f_monotone_increasing : 
  (∀ x ∈ Set.Icc (3 * Real.pi / 4) Real.pi, 
    ∀ y ∈ Set.Icc (3 * Real.pi / 4) Real.pi, 
    x < y → f x < f y) := by
  sorry

-- Define omega and phi to match the original problem conditions
noncomputable def ω : ℝ := 2
noncomputable def φ : ℝ := Real.pi / 3

-- State the conditions from the original problem
axiom omega_positive : ω > 0
axiom phi_bound : abs φ < Real.pi / 2
axiom symmetry_distance : ∀ x : ℝ, f (x + Real.pi / 2) = f x
axiom symmetry_condition : ∀ x : ℝ, f (x + Real.pi / 6) = f (-x)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l786_78662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_to_gym_l786_78632

-- Define constants
def distance_home_to_grocery : ℝ := 720
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define Angelina's speed from home to grocery
noncomputable def speed_home_to_grocery : ℝ → ℝ := λ v => v

-- Define Angelina's speed from grocery to gym
noncomputable def speed_grocery_to_gym : ℝ → ℝ := λ v => 2 * v

-- Define the time taken from home to grocery
noncomputable def time_home_to_grocery : ℝ → ℝ := λ v => distance_home_to_grocery / (speed_home_to_grocery v)

-- Define the time taken from grocery to gym
noncomputable def time_grocery_to_gym : ℝ → ℝ := λ v => distance_grocery_to_gym / (speed_grocery_to_gym v)

-- State the theorem
theorem angelinas_speed_to_gym :
  ∃ v : ℝ, v > 0 ∧ 
  time_home_to_grocery v - time_grocery_to_gym v = time_difference ∧
  speed_grocery_to_gym v = 24 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelinas_speed_to_gym_l786_78632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_surface_area_l786_78624

/-- A cone with an equilateral triangle as its axial section and side length 2 has surface area 3π -/
theorem equilateral_cone_surface_area :
  ∀ (cone : Real → Real → Real),
  (∀ s, cone 2 s = 2) →  -- axial section is equilateral triangle with side length 2
  (∀ r h, cone r h = Real.sqrt (r^2 + h^2)) →  -- definition of cone's slant height
  (∃ r h, cone r h = 2 ∧ r = 1 ∧ h = Real.sqrt 3) →  -- specific dimensions of this cone
  π * (1 + Real.sqrt 3) = 3 * π  -- surface area equals 3π
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_surface_area_l786_78624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_reduction_l786_78635

/-- Represents a single operation of summing digits --/
def digitSumOp (n : ℕ) : ℕ := sorry

/-- Represents a sequence of digit sum operations --/
def digitSumSeq (n : ℕ) (ops : List (ℕ → ℕ)) : ℕ := sorry

/-- Checks if a number is single-digit --/
def isSingleDigit (n : ℕ) : Prop := n < 10

theorem digit_sum_reduction (n : ℕ) : 
  ∃ (ops : List (ℕ → ℕ)), isSingleDigit (digitSumSeq n ops) ∧ ops.length ≤ 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_reduction_l786_78635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_park_tickets_l786_78610

/-- The cost of tickets at an amusement park -/
theorem amusement_park_tickets
  (child_ticket : ℕ) (alexander_child : ℕ) (alexander_adult : ℕ)
  (anna_child : ℕ) (anna_adult : ℕ) (price_difference : ℕ) :
  child_ticket = 600 →
  alexander_child = 2 →
  alexander_adult = 3 →
  anna_child = 3 →
  anna_adult = 2 →
  alexander_child * child_ticket + alexander_adult * (child_ticket + 200) =
    anna_child * child_ticket + anna_adult * (child_ticket + 200) + price_difference →
  price_difference = 200 →
  alexander_child * child_ticket + alexander_adult * (child_ticket + 200) = 3600 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amusement_park_tickets_l786_78610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_on_R_l786_78658

-- Define the vertices of the triangle
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (-1, -6)
def C : ℝ × ℝ := (-3, 2)

-- Define the triangular region R
def R : Set (ℝ × ℝ) := {p | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
  p = (a * A.fst + b * B.fst + c * C.fst, a * A.snd + b * B.snd + c * C.snd)}

-- Define the function f(x, y) = 4x - 3y
def f : ℝ × ℝ → ℝ := λ p ↦ 4 * p.1 - 3 * p.2

theorem extrema_of_f_on_R :
  (∀ p ∈ R, f p ≤ 14) ∧ (∃ p ∈ R, f p = 14) ∧
  (∀ p ∈ R, f p ≥ -18) ∧ (∃ p ∈ R, f p = -18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_on_R_l786_78658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_of_f_l786_78677

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then -2*x - 6 else x/3 + 2

theorem sum_of_roots_of_f (S : Finset ℝ) : 
  (∀ x ∈ S, f x = 0) ∧ (∀ x, f x = 0 → x ∈ S) → S.sum id = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_of_f_l786_78677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_dimensions_l786_78657

/-- The surface area of a sphere with radius r -/
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- The curved surface area of a right circular cylinder with radius r and height h -/
noncomputable def cylinder_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h

theorem cylinder_dimensions (sphere_radius : ℝ) (cyl_radius cyl_height : ℝ) :
  sphere_radius = 5 →
  cyl_height = 2 * cyl_radius →
  sphere_surface_area sphere_radius = cylinder_surface_area cyl_radius cyl_height →
  cyl_height = 10 ∧ 2 * cyl_radius = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_dimensions_l786_78657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acetic_acid_mw_approx_l786_78685

/-- The molecular weight of 7 moles of Acetic acid in grams -/
noncomputable def acetic_acid_7_moles : ℝ := 420

/-- The number of moles of Acetic acid -/
noncomputable def num_moles : ℝ := 7

/-- The molecular weight of 1 mole of Acetic acid in g/mol -/
noncomputable def acetic_acid_molecular_weight : ℝ := acetic_acid_7_moles / num_moles

theorem acetic_acid_mw_approx :
  ∃ ε > 0, |acetic_acid_molecular_weight - 60| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acetic_acid_mw_approx_l786_78685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l786_78615

theorem coefficient_of_x (a : ℝ) : 
  (5 * (-2) + 4 * a = 2) → a = 3 := by
  intro h
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l786_78615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l786_78694

theorem problem_statement (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : (1/a + 1/b) * (1/c + 1/d) + 1/(a*b) + 1/(c*d) = 6/Real.sqrt (a*b*c*d)) :
  (a^2 + a*c + c^2) / (b^2 - b*d + d^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l786_78694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_digit_square_palindromes_l786_78692

/-- A number is a 4-digit number if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is a palindrome if it reads the same forwards and backwards -/
def is_palindrome (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits = digits.reverse

/-- The main theorem stating that there are no 4-digit perfect squares that are palindromes -/
theorem no_four_digit_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit (n^2) ∧ is_palindrome (n^2) :=
by
  sorry

#check no_four_digit_square_palindromes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_digit_square_palindromes_l786_78692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bound_l786_78637

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that under certain conditions, its eccentricity e is bounded. -/
theorem ellipse_eccentricity_bound (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : c^2 = a^2 - b^2) : 
  let e := c / a
  let A := (c, b^2 / a)
  let B := (c, -b^2 / a)
  let F1 := (-c, 0)
  (angle_BAF1 < π/2 ∧ angle_ABF1 < π/2) → 
  Real.sqrt 2 - 1 < e ∧ e < 1 := by
  sorry

-- Define angle_BAF1 and angle_ABF1
def angle_BAF1 (A B F1 : ℝ × ℝ) : ℝ := sorry
def angle_ABF1 (A B F1 : ℝ × ℝ) : ℝ := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_bound_l786_78637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digits_product_5_4_l786_78627

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 →  -- a is a 5-digit number
    1000 ≤ b ∧ b < 10000 →    -- b is a 4-digit number
    a * b < 1000000000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digits_product_5_4_l786_78627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_10th_game_l786_78669

-- Define the scores for games 6 to 9
def scores_6_to_9 : List ℝ := [23, 14, 11, 20]

-- Define the properties of the scores
def avg_increased_after_9 (scores_1_to_5 : List ℝ) : Prop :=
  (scores_1_to_5.sum + scores_6_to_9.sum) / 9 > scores_1_to_5.sum / 5

def avg_after_10_greater_than_18 (scores_1_to_5 : List ℝ) (score_10 : ℝ) : Prop :=
  (scores_1_to_5.sum + scores_6_to_9.sum + score_10) / 10 > 18

-- Define the theorem
theorem min_score_10th_game : 
  ∃ (scores_1_to_5 : List ℝ) (score_10 : ℝ),
    scores_1_to_5.length = 5 ∧ 
    avg_increased_after_9 scores_1_to_5 ∧
    avg_after_10_greater_than_18 scores_1_to_5 score_10 ∧
    score_10 ≥ 29 ∧
    (∀ (score_10' : ℝ), score_10' < 29 → 
      ¬(avg_after_10_greater_than_18 scores_1_to_5 score_10')) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_10th_game_l786_78669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_threes_divisible_by_19_l786_78675

/-- The function that generates the number by inserting k threes between the zeros in 12008 -/
def insert_threes (k : ℕ) : ℕ :=
  120 * 10^(k + 2) + 3 * ((10^(k + 1) - 1) / 9) + 8

/-- Theorem stating that the number formed by inserting any number of threes 
    between the zeros in 12008 is divisible by 19 -/
theorem threes_divisible_by_19 (k : ℕ) : 
  19 ∣ insert_threes k := by
  sorry

#eval insert_threes 0  -- Should output 12008
#eval insert_threes 1  -- Should output 123008
#eval insert_threes 2  -- Should output 123308

end NUMINAMATH_CALUDE_ERRORFEEDBACK_threes_divisible_by_19_l786_78675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l786_78649

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
   (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), f (x - π/6) + f (-x - π/6) = -Real.sqrt 3) ∧
  (∀ (x : ℝ), f (x + 2*π/3) = f (-x + 2*π/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l786_78649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l786_78634

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal

/-- The area of a rhombus -/
noncomputable def Rhombus.area (r : Rhombus) : ℝ := (r.d1 * r.d2) / 2

/-- The length of one side of a rhombus -/
noncomputable def Rhombus.side_length (r : Rhombus) : ℝ :=
  Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)

theorem rhombus_properties (r : Rhombus) (h1 : r.d1 = 18) (h2 : r.d2 = 16) :
  r.area = 144 ∧ r.side_length = Real.sqrt 145 := by
  sorry

#check rhombus_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_properties_l786_78634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l786_78690

/-- The eccentricity of an ellipse with parametric equations x = a * cos(θ) and y = b * sin(θ) -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (min a b / max a b) ^ 2)

/-- The ellipse defined by parametric equations x = 3cos(θ) and y = 4sin(θ) -/
def ellipse_params : ℝ × ℝ := (3, 4)

/-- Theorem: The eccentricity of the ellipse defined by x = 3cos(θ) and y = 4sin(θ) is √7/4 -/
theorem ellipse_eccentricity :
  eccentricity ellipse_params.1 ellipse_params.2 = Real.sqrt 7 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l786_78690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l786_78612

/-- The perimeter of a triangle formed by a point on an ellipse and its foci -/
theorem ellipse_triangle_perimeter (a : ℝ) (h_a : a > Real.sqrt 3) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / 3 = 1}
  let A := (1, (3 : ℝ) / 2)
  let F₁ := (-Real.sqrt (a^2 - 3), 0)
  let F₂ := (Real.sqrt (a^2 - 3), 0)
  A ∈ C → 
  dist A F₁ + dist A F₂ + dist F₁ F₂ = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l786_78612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_X_is_eight_ninths_sin_X_satisfies_properties_l786_78613

/-- Triangle XYZ with specific properties -/
structure TriangleXYZ where
  -- Area of the triangle
  area : ℝ
  -- Sides of the triangle
  xy : ℝ
  xz : ℝ
  -- Geometric mean between XY and XZ
  geom_mean : ℝ
  -- Conditions
  area_eq : area = 100
  geom_mean_eq : geom_mean = 15
  xy_twice_xz : xy = 2 * xz

/-- The sine of angle X in the given triangle -/
noncomputable def sin_X (t : TriangleXYZ) : ℝ := 8 / 9

/-- Theorem stating that sin X is 8/9 for the given triangle -/
theorem sin_X_is_eight_ninths (t : TriangleXYZ) : sin_X t = 8 / 9 := by
  -- Unfold the definition of sin_X
  unfold sin_X
  -- The proof is complete
  rfl

/-- Proof that the sin_X function satisfies the triangle's properties -/
theorem sin_X_satisfies_properties (t : TriangleXYZ) : 
  t.area = (1/2) * t.xy * t.xz * sin_X t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_X_is_eight_ninths_sin_X_satisfies_properties_l786_78613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_extreme_values_l786_78667

/-- The function f(x) = x³ + ax² + 6x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 6*x - 3

/-- f(x) has extreme values on ℝ -/
def has_extreme_values (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂)

theorem min_a_for_extreme_values :
  (∀ a : ℕ, a < 5 → ¬(has_extreme_values (a : ℝ))) ∧
  has_extreme_values 5 := by
  sorry

#check min_a_for_extreme_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_extreme_values_l786_78667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l786_78620

-- Define constants
def platform_length : ℝ := 30.0024
def platform_passing_time : ℝ := 22
def train_speed_kmh : ℝ := 54

-- Define functions
noncomputable def km_per_hour_to_m_per_second (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def calculate_train_length (platform_length platform_passing_time train_speed_ms : ℝ) : ℝ :=
  train_speed_ms * platform_passing_time - platform_length

noncomputable def calculate_man_passing_time (train_length train_speed_ms : ℝ) : ℝ :=
  train_length / train_speed_ms

-- Theorem statement
theorem train_passing_man_time :
  let train_speed_ms := km_per_hour_to_m_per_second train_speed_kmh
  let train_length := calculate_train_length platform_length platform_passing_time train_speed_ms
  let man_passing_time := calculate_man_passing_time train_length train_speed_ms
  ∃ ε > 0, |man_passing_time - 20| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l786_78620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_l786_78619

/-- An ellipse with specified points -/
structure Ellipse where
  A : ℝ × ℝ  -- endpoint of major axis
  B : ℝ × ℝ  -- endpoint of minor axis
  F₁ : ℝ × ℝ  -- focus 1
  F₂ : ℝ × ℝ  -- focus 2

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point to point -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

/-- Theorem: If AF₁ · AF₂ + BF₁ · BF₂ = 0 in an ellipse, then |AB| / |F₁F₂| = √2 / 2 -/
theorem ellipse_ratio (Γ : Ellipse) 
  (h : dot_product (vector Γ.A Γ.F₁) (vector Γ.A Γ.F₂) + 
       dot_product (vector Γ.B Γ.F₁) (vector Γ.B Γ.F₂) = 0) : 
  distance Γ.A Γ.B / distance Γ.F₁ Γ.F₂ = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_l786_78619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_season_length_l786_78606

/-- The number of football games in a certain month -/
noncomputable def games_per_month : ℝ := 323.0

/-- The total number of football games played in the season -/
noncomputable def total_games : ℝ := 5491.0

/-- The number of months in the season -/
noncomputable def season_months : ℝ := total_games / games_per_month

/-- Theorem stating that the number of months in the season is 17.0 -/
theorem season_length : season_months = 17.0 := by
  -- Unfold the definitions
  unfold season_months total_games games_per_month
  -- Perform the division
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_season_length_l786_78606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_median_length_l786_78661

/-- Represents an isosceles trapezoid ABCD with median EF and height CG -/
structure IsoscelesTrapezoid where
  /-- Length of the diagonal AC -/
  diagonal : ℝ
  /-- Length of the height CG -/
  height : ℝ
  /-- AD is parallel to BC -/
  parallel_bases : Prop

/-- The length of the median in an isosceles trapezoid -/
noncomputable def median_length (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt (t.diagonal^2 - t.height^2)

/-- Theorem: In an isosceles trapezoid with diagonal 25 and height 15, the median length is 20 -/
theorem isosceles_trapezoid_median_length :
  let t : IsoscelesTrapezoid := { diagonal := 25, height := 15, parallel_bases := True }
  median_length t = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_median_length_l786_78661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l786_78631

theorem inequality_proof (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  let a := (m^(m+1) + n^(n+1)) / (m^m + n^n : ℝ)
  a^m + a^n ≥ m^m + n^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l786_78631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_position_l786_78693

theorem sequence_term_position :
  ∃ n : ℕ, (fun k : ℕ => (2 : ℚ) / (k^2 + k : ℚ)) n = 1/10 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_position_l786_78693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_exists_l786_78628

/-- A system of equations has a solution for some b if and only if a is in the specified set -/
theorem system_solution_exists (a : ℝ) : 
  (∃ (b x y : ℝ), 
    x = 6 / a - abs (y - a) ∧ 
    x^2 + y^2 + b^2 + 63 = 2*(b*y - 8*x)) ↔ 
  (a ≤ -2/3 ∨ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_exists_l786_78628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l786_78646

noncomputable def angle (a b : ℝ × ℝ) : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_angle_problem (a b : ℝ × ℝ) : 
  a = (1, -Real.sqrt 3) → 
  Real.sqrt (b.1^2 + b.2^2) = 1 → 
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 → 
  angle a b = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l786_78646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l786_78616

-- Define the polynomial type
def MyPolynomial (α : Type*) := ℕ → α

-- Define the polynomial subtraction operation
def polySub {α : Type*} [Ring α] (p q : MyPolynomial α) : MyPolynomial α :=
  fun n => p n - q n

-- Define the polynomial evaluation
def polyEval {α : Type*} [Ring α] (p : MyPolynomial α) (x y : α) : α :=
  p 2 * x^2 * y + p 1 * x * y + p 0

-- State the theorem
theorem polynomial_problem {α : Type*} [CommRing α] (P : MyPolynomial α) :
  (polyEval (polySub P (fun _ => -1)) = fun x y => 3*x^2*y - 2*x*y - 1) →
  (polyEval P = fun x y => 2*x^2*y - 2*x*y - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_problem_l786_78616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_is_sqrt_2_l786_78654

/-- The line l in parametric form -/
def line_l (t : ℝ) : ℝ × ℝ := (t, t + 1)

/-- The circle C in polar form -/
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

/-- The distance from the center of circle C to line l -/
noncomputable def distance_center_to_line : ℝ := Real.sqrt 2

theorem distance_center_to_line_is_sqrt_2 :
  let center : ℝ × ℝ := (1, 0)
  let line_equation (x y : ℝ) : ℝ := x - y + 1
  distance_center_to_line = 
    (abs (line_equation center.1 center.2)) / 
    (Real.sqrt ((1 : ℝ)^2 + 1^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_is_sqrt_2_l786_78654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washer_cost_is_4_l786_78655

/-- The cost of a washer at a laundromat -/
def washer_cost : ℚ := 0  -- Initialize with a default value

/-- The cost of running a dryer for 10 minutes -/
def dryer_cost_per_10_min : ℚ := 1/4

/-- The number of loads Samantha washes -/
def loads_washed : ℕ := 2

/-- The number of dryers Samantha uses -/
def dryers_used : ℕ := 3

/-- The time each dryer runs in minutes -/
def dryer_time : ℕ := 40

/-- The total cost Samantha spends -/
def total_cost : ℚ := 11

/-- Theorem stating that the cost of a washer is $4 -/
theorem washer_cost_is_4 :
  washer_cost = 4 :=
by
  sorry

#eval washer_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_washer_cost_is_4_l786_78655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l786_78640

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ x^2 + 2*m*x + 1 = 0 ∧ y^2 + 2*m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*(m - 2)*x - 3*m + 10 ≠ 0

theorem range_of_m : {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)} = Set.Iic (-2) ∪ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l786_78640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_equivalent_terminal_side_l786_78678

theorem cosine_of_equivalent_terminal_side (α β : Real) : 
  α = -1035 * π / 180 → 
  (∃ k : ℤ, β = α + k * (2 * π)) → 
  Real.cos β = Real.sqrt 2 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_equivalent_terminal_side_l786_78678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l786_78673

/-- The parabola y^2 = 3x with focus F -/
noncomputable def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 3 * p.1}

/-- The focus of the parabola -/
noncomputable def F : ℝ × ℝ := (3/4, 0)

/-- A line passing through F with an inclination angle of 30° -/
noncomputable def line (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * (x - 3/4)

/-- Points A and B where the line intersects the parabola -/
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

/-- The theorem to be proved -/
theorem length_AB : ‖A - B‖ = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l786_78673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_interpretation_l786_78656

theorem correlation_coefficient_interpretation 
  (R : ℝ) -- Correlation coefficient between height and weight
  (h : R^2 = 0.64) -- Given condition
  : ∃ (p : ℝ), p = 64 ∧ p = 100 * R^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_coefficient_interpretation_l786_78656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l786_78682

-- Define the triangle ABC
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- State the theorem
theorem triangle_side_and_area 
  (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 3 * Real.sqrt 3) 
  (h_c : c = 2) 
  (h_B : B = 5 * Real.pi / 6) -- 150° in radians
  (h_triangle : triangle a b c A B C) :
  b = 7 ∧ (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l786_78682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_range_l786_78607

/-- The ellipse C -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- A point is on the ellipse C -/
def on_ellipse (p : ℝ × ℝ) : Prop := p ∈ C

/-- Vector addition -/
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

/-- Scalar multiplication of a vector -/
def vec_smul (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (t * v.1, t * v.2)

/-- Vector subtraction -/
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Magnitude of a vector -/
noncomputable def vec_mag (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem ellipse_t_range (A B P : ℝ × ℝ) (t : ℝ) 
  (hA : on_ellipse A) (hB : on_ellipse B) (hP : on_ellipse P)
  (h_vec : vec_add A B = vec_smul t P)
  (h_ineq : vec_mag (vec_sub P A) - vec_mag (vec_sub P B) < 2 * Real.sqrt 5 / 3) :
  t ∈ Set.Ioo (-2 : ℝ) (-2 * Real.sqrt 6 / 3) ∪ Set.Ioo (2 * Real.sqrt 6 / 3) 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_range_l786_78607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l786_78696

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 - 6*x + y^2 + 8 = 0

-- State the theorem
theorem min_tangent_length : 
  ∃ (x y : ℝ), line x y ∧ my_circle x y ∧
  (∀ (x' y' : ℝ), line x' y' ∧ my_circle x' y' → 
    (x - x')^2 + (y - y')^2 ≥ 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l786_78696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_sums_interval_iff_condition_l786_78687

/-- A sequence of monotonically decreasing positive terms that converges -/
def DecreasingConvergentSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, a (n + 1) ≤ a n) ∧ 
  Summable a

/-- The set of all sums of subsequences -/
def SubsequenceSums (a : ℕ → ℝ) : Set ℝ :=
  {s | ∃ (b : ℕ → ℝ), (∀ n, ∃ m, b n = a m) ∧ Summable b ∧ s = ∑' n, b n}

/-- The condition for the set of subsequence sums to be an interval -/
def IntervalCondition (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≤ ∑' i, a (i + n)

theorem subsequence_sums_interval_iff_condition 
  (a : ℕ → ℝ) (h : DecreasingConvergentSequence a) :
  (∃ l u, SubsequenceSums a = Set.Icc l u) ↔ IntervalCondition a :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_sums_interval_iff_condition_l786_78687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_240_l786_78605

/-- The length of a train given its speed, platform length, and time to cross the platform -/
noncomputable def train_length (speed : ℝ) (platform_length : ℝ) (time : ℝ) : ℝ :=
  speed * (5/18) * time - platform_length

/-- Theorem stating that the train length is 240 meters under given conditions -/
theorem train_length_is_240 :
  train_length 72 280 26 = 240 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Simplify the arithmetic expression
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_240_l786_78605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_line_equation_l786_78625

/-- A line that passes through point (2,1) and forms an isosceles right triangle with the coordinate axes -/
structure IsoscelesRightTriangleLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (2,1) -/
  passes_through_point : slope * 2 + y_intercept = 1
  /-- The line forms an isosceles right triangle with the coordinate axes -/
  forms_isosceles_right_triangle : 
    (slope > 0 ∧ y_intercept = -slope * y_intercept) ∨
    (slope < 0 ∧ y_intercept = slope * y_intercept)

/-- The equation of the line in the form ax + by + c = 0 -/
noncomputable def line_equation (l : IsoscelesRightTriangleLine) : ℝ × ℝ × ℝ :=
  if l.slope > 0 then
    (1, 1, -3)
  else
    (1, -1, -1)

theorem isosceles_right_triangle_line_equation (l : IsoscelesRightTriangleLine) :
  line_equation l = (1, 1, -3) ∨ line_equation l = (1, -1, -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_line_equation_l786_78625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_find_a_b_l786_78671

-- Define the complex number z
noncomputable def z : ℂ := ((1 + Complex.I)^2 + 2*(5 - Complex.I)) / (3 + Complex.I)

-- Theorem for |z|
theorem abs_z : Complex.abs z = Real.sqrt 10 := by sorry

-- Theorem for a and b
theorem find_a_b (a b : ℝ) (h : z * (z + a) = b + Complex.I) : a = -7 ∧ b = -13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_find_a_b_l786_78671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_staff_for_security_check_l786_78686

/-- Represents the security check scenario at the Beijing Olympics -/
structure SecurityCheck where
  /-- Number of people checked per minute per staff member -/
  efficiency : ℝ
  /-- Number of spectators arriving per minute -/
  arrival_rate : ℝ
  /-- Initial number of spectators waiting -/
  initial_waiting : ℝ

/-- Calculates the time needed for a given number of staff members -/
noncomputable def time_needed (sc : SecurityCheck) (staff : ℝ) : ℝ :=
  sc.initial_waiting / (staff * sc.efficiency - sc.arrival_rate)

/-- Theorem stating the minimum number of staff needed for the security check -/
theorem min_staff_for_security_check (sc : SecurityCheck) :
  (time_needed sc 3 = 25) →
  (time_needed sc 6 = 10) →
  (∀ m : ℝ, m < 11 → time_needed sc m > 5) ∧
  (time_needed sc 11 ≤ 5) := by
  sorry

#check min_staff_for_security_check

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_staff_for_security_check_l786_78686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_600_degrees_l786_78604

/-- Represents a quadrant in the coordinate plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines the quadrant of an angle in degrees -/
noncomputable def angleToQuadrant (angle : ℝ) : Quadrant :=
  let normalizedAngle := angle % 360
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then Quadrant.first
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then Quadrant.second
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then Quadrant.third
  else Quadrant.fourth

theorem terminal_side_600_degrees :
  angleToQuadrant 600 = Quadrant.third := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_600_degrees_l786_78604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_place_team_draws_prove_first_place_draws_l786_78639

/-- Represents a soccer team's performance in a tournament -/
structure TeamPerformance where
  wins : ℕ
  draws : ℕ

/-- Calculates the total points for a team given their performance -/
def calculate_points (team : TeamPerformance) : ℕ :=
  team.wins * 3 + team.draws

/-- Represents the tournament results -/
structure TournamentResults where
  joe_team : TeamPerformance
  first_place_team : TeamPerformance
  point_difference : ℕ

/-- Theorem stating that the first-place team drew 2 games -/
theorem first_place_team_draws (results : TournamentResults) : 
  results.first_place_team.draws = 2 :=
  by
    sorry

/-- Main theorem: Given the tournament conditions, prove that the first-place team drew 2 games -/
theorem prove_first_place_draws : ∃ (results : TournamentResults), 
  results.joe_team = ⟨1, 3⟩ ∧
  results.first_place_team.wins = 2 ∧
  results.point_difference = 2 ∧
  results.first_place_team.draws = 2 :=
  by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_place_team_draws_prove_first_place_draws_l786_78639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_normal_at_t₀_l786_78633

/-- Parametric curve definition -/
noncomputable def curve (t : ℝ) : ℝ × ℝ :=
  (1 - t^2, t - t^3)

/-- Tangent line at t₀ -/
noncomputable def tangent_line (t₀ : ℝ) (x : ℝ) : ℝ :=
  (11/4) * x + 9/4

/-- Normal line at t₀ -/
noncomputable def normal_line (t₀ : ℝ) (x : ℝ) : ℝ :=
  -(4/11) * x - 78/11

theorem curve_tangent_normal_at_t₀ (t₀ : ℝ) (h : t₀ = 2) :
  let (x₀, y₀) := curve t₀
  (∀ x, tangent_line t₀ x = (11/4) * x + 9/4) ∧
  (∀ x, normal_line t₀ x = -(4/11) * x - 78/11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_normal_at_t₀_l786_78633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l786_78684

-- Define set A
def A : Set ℝ := {x : ℝ | x ≤ -2 ∨ x > 1}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := Set.Ioo (2*a - 3) (a + 1)

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∪ B a = Set.univ → 0 < a ∧ a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l786_78684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_theorem_l786_78689

/-- The side length of a cube given specific conditions with unit spheres -/
noncomputable def cube_side_length : ℚ := 2/3

/-- Four unit spheres placed on a horizontal plane with centers forming a square -/
def sphere_centers_form_square (centers : Fin 4 → ℝ × ℝ) : Prop :=
  ∀ i j : Fin 4, i ≠ j → Real.sqrt ((centers i).1 - (centers j).1)^2 + ((centers i).2 - (centers j).2)^2 = 2

/-- The cube's bottom face is on the same plane as the spheres -/
def cube_on_plane (cube_base : ℝ × ℝ) (side_length : ℚ) : Prop :=
  True  -- This is always true given the problem statement

/-- The top vertices of the cube each touch one of the four spheres -/
def cube_touches_spheres (cube_base : ℝ × ℝ) (side_length : ℚ) (centers : Fin 4 → ℝ × ℝ) : Prop :=
  ∀ i : Fin 4, ∃ v : ℝ × ℝ, 
    Real.sqrt ((v.1 - cube_base.1)^2 + (v.2 - cube_base.2)^2) = Real.sqrt 2 * (side_length : ℝ) ∧
    Real.sqrt ((v.1 - (centers i).1)^2 + (v.2 - (centers i).2)^2) = 1

theorem cube_side_length_theorem (centers : Fin 4 → ℝ × ℝ) (cube_base : ℝ × ℝ) :
  sphere_centers_form_square centers →
  cube_on_plane cube_base cube_side_length →
  cube_touches_spheres cube_base cube_side_length centers →
  cube_side_length = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_side_length_theorem_l786_78689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l786_78617

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

noncomputable def g (x : ℝ) := f (x + Real.pi/4) + f (x + 3*Real.pi/4)

theorem f_and_g_properties :
  (f (Real.pi/2) = 1) ∧
  (∀ (p : ℝ), p > 0 → (∀ (x : ℝ), f (x + p) = f x) → p ≥ 2*Real.pi) ∧
  (∀ (x : ℝ), g x ≥ -2) ∧
  (∃ (x : ℝ), g x = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l786_78617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_dividing_polygon_equally_l786_78652

-- Define the polygon OABCDE
def polygon_vertices : List (ℝ × ℝ) := [(0,0), (0,6), (4,6), (4,4), (6,4), (6,0)]

-- Define the point M
def point_M : ℝ × ℝ := (2,3)

-- Function to calculate the area of a polygon given its vertices
noncomputable def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Function to check if a line divides a polygon into two equal areas
def divides_equally (line : ℝ → ℝ) (vertices : List (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem line_dividing_polygon_equally :
  ∃ m c : ℝ, 
    (∀ x, (λ x => m * x + c) x = -1/3 * x + 11/3) ∧
    (m * point_M.fst + c = point_M.snd) ∧
    divides_equally (λ x => m * x + c) polygon_vertices := by
  sorry

#check line_dividing_polygon_equally

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_dividing_polygon_equally_l786_78652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_triangle_perimeter_l786_78698

open Real

noncomputable def f (x : ℝ) : ℝ := sin x + Real.sqrt 3 * cos x

theorem vector_parallel_and_triangle_perimeter 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h_parallel : ∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) 
  (h_a : a = (1/2, 1/2 * sin x + Real.sqrt 3/2 * cos x)) 
  (h_b : b = (1, f x)) 
  (A : ℝ) 
  (h_acute : 0 < A ∧ A < π/2) 
  (h_f : f (A - π/3) = Real.sqrt 3) 
  (BC : ℝ) 
  (h_BC : BC = Real.sqrt 3) :
  (∀ x, f x = sin x + Real.sqrt 3 * cos x) ∧ 
  (∃ (P : ℝ), P ≤ 3 * Real.sqrt 3 ∧ 
    ∀ (AB AC : ℝ), 
      0 < AB ∧ 0 < AC → 
      AB + AC + BC ≤ P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_and_triangle_perimeter_l786_78698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_N_intersect_M_B_value_P_single_element_l786_78699

-- Define the universal set U
def U : Set ℤ := {x | -5 ≤ x ∧ x ≤ 10}

-- Define set M
def M : Set ℤ := {x | 0 ≤ x ∧ x ≤ 7}

-- Define set N
def N : Set ℤ := {x | -2 ≤ x ∧ x < 4}

-- Theorem 1
theorem complement_N_intersect_M : (U \ N) ∩ M = {4, 5, 6, 7} := by sorry

-- Define the universal set U for the second problem
def U2 : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define sets A and B
variable (A B : Set ℕ)

-- Axiom: U2 is the union of A and B
axiom U2_eq_A_union_B : U2 = A ∪ B

-- Axiom: The intersection of A and complement of B in U2 is {2,4,6,8}
axiom A_intersect_complement_B : A ∩ (U2 \ B) = {2, 4, 6, 8}

-- Theorem 2
theorem B_value : B = {0, 1, 3, 5, 7, 9, 10} := by sorry

-- Define the set P
def P (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * a * x + 1 = 0}

-- Theorem 3
theorem P_single_element :
  ∃ (a : ℝ), (∃! x, x ∈ P a) ∧ a = 1 ∧ (∀ x, x ∈ P a → x = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_N_intersect_M_B_value_P_single_element_l786_78699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchases_for_given_conditions_l786_78663

/-- The optimal number of purchases to minimize total annual cost -/
noncomputable def optimal_purchases (total_tons : ℝ) (freight_cost : ℝ) (storage_cost_factor : ℝ) : ℝ :=
  Real.sqrt ((total_tons * freight_cost) / storage_cost_factor)

theorem optimal_purchases_for_given_conditions :
  optimal_purchases 200 20000 10000 = 10 := by
  -- Unfold the definition of optimal_purchases
  unfold optimal_purchases
  -- Simplify the expression
  simp
  -- The proof is completed using 'sorry' as requested
  sorry

#check optimal_purchases_for_given_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchases_for_given_conditions_l786_78663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_number_is_odd_l786_78691

def operation (numbers : List Int) : List Int :=
  match numbers with
  | a :: b :: rest => (Int.natAbs (a - b)) :: rest
  | _ => numbers

partial def final_number (initial_numbers : List Int) : Int :=
  match initial_numbers with
  | [n] => n
  | _ :: _ => final_number (operation initial_numbers)
  | [] => 0

theorem last_number_is_odd :
  let initial_numbers := List.range 2018
  Odd (final_number (initial_numbers.map Int.ofNat)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_number_is_odd_l786_78691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l786_78664

/-- A set of points on the unit circle with integer angles -/
def K : Set (ℝ × ℝ) :=
  {p | ∃ n : ℤ, p.1 = Real.cos (n : ℝ) ∧ p.2 = Real.sin (n : ℝ)}

/-- A line through the origin and a point on the unit circle -/
def symmetryAxis (x : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = (t * Real.cos x, t * Real.sin x)}

/-- The set of all symmetry axes for K -/
def symmetryAxes : Set (Set (ℝ × ℝ)) :=
  {l | ∃ n : ℤ, l = symmetryAxis (n : ℝ)}

/-- Reflection of a point over a line -/
noncomputable def reflect_over_line (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Reflection of a point through another point -/
noncomputable def reflect_through_point (c : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

theorem existence_of_special_set :
  ∃ (S : Set (ℝ × ℝ)),
    (∃ (r : ℝ), ∀ p ∈ S, p.1^2 + p.2^2 ≤ r^2) ∧  -- S is bounded
    (¬ Set.Countable {l | ∀ p ∈ S, p ∈ l → reflect_over_line l p ∈ S}) ∧  -- S has uncountably many symmetry axes
    (∀ c : ℝ × ℝ, ∃ p ∈ S, reflect_through_point c p ∉ S)  -- S is not centrally symmetric
    := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l786_78664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l786_78643

theorem garden_area_increase :
  ∀ (original_length original_width : ℝ),
    original_length = 50 ∧
    original_width = 10 →
    (let original_perimeter := 2 * (original_length + original_width)
     let original_area := original_length * original_width
     let new_side_length := original_perimeter / 4
     let new_area := new_side_length * new_side_length
     new_area - original_area) = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l786_78643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccerStoreDiscountRatio_l786_78626

/-- Calculates the lowest possible sale price for an item given its list price, maximum discount, and additional summer sale discount. -/
def lowestSalePrice (listPrice : ℝ) (maxDiscount : ℝ) (summerDiscount : ℝ) : ℝ :=
  listPrice * (1 - maxDiscount) * (1 - summerDiscount)

/-- Represents the problem of calculating the ratio of the combined lowest possible sale price to the total list price. -/
theorem soccerStoreDiscountRatio :
  let jerseyPrice := 80
  let soccerBallPrice := 40
  let soccerCleatsPrice := 100
  let jerseyMaxDiscount := 0.5
  let soccerBallMaxDiscount := 0.6
  let soccerCleatsMaxDiscount := 0.4
  let summerDiscount := 0.2
  
  let totalListPrice := jerseyPrice + soccerBallPrice + soccerCleatsPrice
  let totalSalePrice := 
    lowestSalePrice jerseyPrice jerseyMaxDiscount summerDiscount +
    lowestSalePrice soccerBallPrice soccerBallMaxDiscount summerDiscount +
    lowestSalePrice soccerCleatsPrice soccerCleatsMaxDiscount summerDiscount
  
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((totalSalePrice / totalListPrice) * 100 - 32.73) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccerStoreDiscountRatio_l786_78626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l786_78614

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the point P
def point_P : ℝ × ℝ := (0, 1)

-- Define a line passing through P
noncomputable def line_through_P (α : ℝ) (t : ℝ) : ℝ × ℝ := (t * Real.cos α, 1 + t * Real.sin α)

-- Define the intersection points A and B
def intersection_points (α : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_through_P α t ∧ curve_C p.1 p.2}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_product_range :
  ∀ α : ℝ, ∀ A B, A ∈ intersection_points α → B ∈ intersection_points α →
    2 * Real.sqrt 3 ≤ distance point_P A * distance point_P B ∧
    distance point_P A * distance point_P B ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l786_78614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_speed_theorem_problem_solution_l786_78603

/-- Represents a train with its travel distance and time -/
structure Train where
  distance : ℝ
  time : ℝ

/-- Calculates the combined average speed of multiple trains -/
noncomputable def combinedAverageSpeed (trains : List Train) : ℝ :=
  let totalDistance := trains.map (λ t => t.distance) |>.sum
  let totalTime := trains.map (λ t => t.time) |>.sum
  totalDistance / totalTime

/-- Theorem: The combined average speed of three trains is equal to the total distance divided by the total time -/
theorem combined_average_speed_theorem (trainA trainB trainC : Train) :
  combinedAverageSpeed [trainA, trainB, trainC] =
  (trainA.distance + trainB.distance + trainC.distance) / (trainA.time + trainB.time + trainC.time) := by
  sorry

/-- Given problem instance -/
def problem_instance : List Train := [
  { distance := 250, time := 4 },
  { distance := 480, time := 6 },
  { distance := 390, time := 5 }
]

/-- Theorem: The combined average speed for the given problem instance is approximately 74.67 km/h -/
theorem problem_solution :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |combinedAverageSpeed problem_instance - 74.67| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_speed_theorem_problem_solution_l786_78603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_is_552_l786_78611

/-- Represents a cube with six faces, each marked with a number. -/
structure Cube where
  faces : Fin 6 → ℕ
  valid_faces : ∀ i, faces i ∈ ({3, 6, 12, 24, 48, 96} : Set ℕ)

/-- Represents a stack of three cubes. -/
structure CubeStack where
  bottom : Cube
  middle : Cube
  top : Cube

/-- Calculates the sum of visible numbers on a cube stack. -/
def visible_sum (stack : CubeStack) : ℕ :=
  sorry

/-- Theorem stating that the maximum possible sum of visible numbers is 552. -/
theorem max_visible_sum_is_552 :
  ∃ (stack : CubeStack), ∀ (other_stack : CubeStack),
    visible_sum stack ≥ visible_sum other_stack ∧
    visible_sum stack = 552 :=
  sorry

#check max_visible_sum_is_552

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_is_552_l786_78611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_minus_x_squared_minus_x_l786_78695

theorem integral_sqrt_2x_minus_x_squared_minus_x : 
  ∫ x in Set.Icc 0 1, (Real.sqrt (2*x - x^2) - x) = (Real.pi - 2) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2x_minus_x_squared_minus_x_l786_78695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_plus_a_2014_eq_seven_sixths_l786_78630

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1/2 then x + 1/2
  else if 1/2 < x ∧ x < 1 then 2*x - 1
  else x - 1

noncomputable def a : ℕ → ℝ
  | 0 => 7/3
  | n + 1 => f (a n)

theorem a_2013_plus_a_2014_eq_seven_sixths :
  a 2013 + a 2014 = 7/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_plus_a_2014_eq_seven_sixths_l786_78630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_T_l786_78668

def is_valid_set (T : Finset ℕ) : Prop :=
  T.card = 5 ∧
  (∀ x, x ∈ T → 1 ≤ x ∧ x ≤ 15) ∧
  (∀ c d, c ∈ T → d ∈ T → c < d → d ≠ c * c)

theorem least_element_in_T :
  ∃ T : Finset ℕ, is_valid_set T ∧
  (∀ x, x ∈ T → 2 ≤ x) ∧
  (∀ S : Finset ℕ, is_valid_set S → ∃ y, y ∈ S ∧ 2 ≤ y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_T_l786_78668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_path_with_untouched_face_l786_78697

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (h_size : size = 8)

/-- Represents a die --/
structure Die :=
  (faces : Finset Nat)
  (h_faces : faces.card = 6)

/-- Represents a path on the chessboard --/
def ChessPath (n : Nat) := Fin n → Fin 8 × Fin 8

/-- Predicate to check if a path covers all squares of the chessboard --/
def covers_all_squares (p : ChessPath n) : Prop :=
  ∀ (i j : Fin 8), ∃ (k : Fin n), p k = (i, j)

/-- Represents the bottom face of a die at a given position --/
def bottom_face (d : Die) (pos : Fin 8 × Fin 8) : Nat :=
  sorry -- Implementation details omitted

/-- Predicate to check if a face never touches the board during a roll --/
def face_never_touches (d : Die) (p : ChessPath n) (f : Nat) : Prop :=
  f ∈ d.faces ∧ ∀ (k : Fin n), f ≠ bottom_face d (p k)

/-- The main theorem --/
theorem exists_path_with_untouched_face (b : Chessboard) (d : Die) :
  ∃ (n : Nat) (p : ChessPath n) (f : Nat),
    covers_all_squares p ∧ face_never_touches d p f :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_path_with_untouched_face_l786_78697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_l786_78636

/-- The rate at which a loom weaves cloth, given the amount of cloth woven and the time taken. -/
noncomputable def weaving_rate (cloth_woven : ℝ) (time_taken : ℝ) : ℝ :=
  cloth_woven / time_taken

/-- Theorem stating that the loom weaves approximately 0.128 meters of cloth per second. -/
theorem loom_weaving_rate : 
  ∃ ε > 0, |weaving_rate 27 210.9375 - 0.128| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loom_weaving_rate_l786_78636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_distance_approx_l786_78653

/-- Calculates the downhill distance given uphill and downhill speeds, uphill distance, and average speed -/
noncomputable def calculate_downhill_distance (uphill_speed downhill_speed uphill_distance average_speed : ℝ) : ℝ :=
  let total_time := uphill_distance / uphill_speed + (average_speed * (uphill_distance / uphill_speed) - uphill_distance) / (downhill_speed - average_speed)
  (average_speed * total_time - uphill_distance)

/-- Theorem stating that given the specified conditions, the downhill distance is approximately 49.96 km -/
theorem downhill_distance_approx (ε : ℝ) (hε : ε > 0) : 
  ∃ (D : ℝ), 
    calculate_downhill_distance 30 80 100 37.89 = D ∧ 
    abs (D - 49.96) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downhill_distance_approx_l786_78653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l786_78665

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The left focus of the hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ := -Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: If the right vertex of a hyperbola lies inside the circle with diameter AB,
    where A and B are the intersections of perpendiculars from the left focus to the x-axis
    with the hyperbola, then the eccentricity of the hyperbola is greater than 2 -/
theorem hyperbola_eccentricity_bound (h : Hyperbola) 
    (vertex_inside : ∃ (A B : ℝ × ℝ), 
      A.1 = left_focus h ∧ 
      B.1 = left_focus h ∧
      (A.1 - h.a)^2 / h.a^2 - A.2^2 / h.b^2 = 1 ∧
      (B.1 - h.a)^2 / h.a^2 - B.2^2 / h.b^2 = 1 ∧
      h.a < (A.1 + B.1) / 2 + Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2) :
  eccentricity h > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l786_78665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zeros_in_intervals_l786_78601

noncomputable def f (x : ℝ) := (1/3) * x - Real.log x

theorem f_has_zeros_in_intervals :
  (∃ x₁ ∈ Set.Ioo 0 3, f x₁ = 0) ∧
  (∃ x₂ ∈ Set.Ioi 3, f x₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zeros_in_intervals_l786_78601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l786_78600

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy }

/-- The x-coordinate of the intersection point of a line with the x-axis -/
noncomputable def xAxisIntersection (l : Line) : ℝ :=
  -l.intercept / l.slope

/-- The original line y = 2x - 4 -/
def originalLine : Line :=
  { slope := 2, intercept := -4 }

/-- The amount of upward translation -/
def translationAmount : ℝ := 2

theorem intersection_point_coordinates :
  let translatedLine := translateLine originalLine translationAmount
  xAxisIntersection translatedLine = 1 ∧
  0 = translatedLine.slope * (xAxisIntersection translatedLine) + translatedLine.intercept := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l786_78600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xi_2_l786_78642

/-- A random variable ξ that takes values 1, 2, and 3 -/
def ξ : ℕ → ℝ := sorry

/-- The parameter a in the probability mass function -/
noncomputable def a : ℝ := 3

/-- The probability mass function for ξ -/
noncomputable def P (i : ℕ) : ℝ := i / (2 * a)

/-- The sum of probabilities equals 1 -/
axiom prob_sum : P 1 + P 2 + P 3 = 1

/-- The probability of ξ = 2 is 1/3 -/
theorem prob_xi_2 : P 2 = 1/3 := by
  -- Expand the definition of P
  unfold P
  -- Substitute the value of a
  simp [a]
  -- Simplify the fraction
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_xi_2_l786_78642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_l786_78672

/-- Given function f(x) = x^2 + ax + 1/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1/x

/-- The interval (1/2, +∞) -/
def interval : Set ℝ := {x | x > 1/2}

/-- f is increasing on the interval -/
def is_increasing (a : ℝ) : Prop :=
  ∀ x y, x ∈ interval → y ∈ interval → x < y → f a x < f a y

/-- The theorem stating the condition for a -/
theorem f_increasing_condition :
  ∀ a : ℝ, is_increasing a ↔ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_l786_78672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_expansion_l786_78609

/-- Legendre polynomials -/
noncomputable def P₀ : ℝ → ℝ := λ _ => 1
noncomputable def P₁ : ℝ → ℝ := λ x => x
noncomputable def P₂ : ℝ → ℝ := λ x => (3 * x^2 - 1) / 2

/-- Fourier series expansion theorem -/
theorem fourier_series_expansion 
  (α β γ : ℝ) (θ : ℝ) (h : 0 < θ ∧ θ < π) :
  α + β * Real.cos θ + γ * (Real.cos θ)^2 = 
    (α + 1/3 * γ) * P₀ (Real.cos θ) + β * P₁ (Real.cos θ) + 2/3 * γ * P₂ (Real.cos θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_expansion_l786_78609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l786_78648

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2 * x - Real.pi / 6) + 2 * (Real.sin (x - Real.pi / 12))^2

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l786_78648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_muslim_boys_l786_78629

theorem percentage_of_muslim_boys
  (total_boys : ℕ)
  (hindu_percentage : ℚ)
  (sikh_percentage : ℚ)
  (other_boys : ℕ)
  (h1 : total_boys = 850)
  (h2 : hindu_percentage = 14 / 100)
  (h3 : sikh_percentage = 10 / 100)
  (h4 : other_boys = 272)
  : (total_boys - ⌊hindu_percentage * total_boys⌋
    - ⌊sikh_percentage * total_boys⌋ - other_boys : ℚ)
    / total_boys * 100 = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_muslim_boys_l786_78629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_distance_is_five_l786_78659

-- Define the walking speeds
noncomputable def heather_speed : ℝ := 5
noncomputable def stacy_speed : ℝ := heather_speed + 1

-- Define the time difference in hours
noncomputable def time_difference : ℝ := 24 / 60

-- Define Heather's distance walked when they meet
noncomputable def heather_distance : ℝ := 1.1818181818181817

-- Theorem to prove
theorem original_distance_is_five :
  let t : ℝ := heather_distance / heather_speed
  let stacy_distance : ℝ := stacy_speed * (t + time_difference)
  heather_distance + stacy_distance = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_distance_is_five_l786_78659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piravena_trip_cost_l786_78641

/-- Represents the cost calculation for a trip between cities -/
noncomputable def TripCost (distance : ℝ) (busCostPerKm : ℝ) (planeCostPerKm : ℝ) (planeBookingFee : ℝ) : ℝ :=
  min (distance * busCostPerKm) (distance * planeCostPerKm + planeBookingFee)

/-- Calculates the total cost of Piravena's trip -/
noncomputable def totalTripCost (AB BC CA : ℝ) (busCostPerKm planeCostPerKm planeBookingFee : ℝ) : ℝ :=
  TripCost AB busCostPerKm planeCostPerKm planeBookingFee +
  TripCost BC busCostPerKm planeCostPerKm (planeBookingFee / 2) +
  TripCost CA busCostPerKm planeCostPerKm (planeBookingFee / 2)

theorem piravena_trip_cost :
  ∀ (AB BC CA : ℝ),
  AB = 4000 →
  CA = 3500 →
  AB^2 = BC^2 + CA^2 →
  totalTripCost AB BC CA 0.18 0.12 120 = 1372.38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_piravena_trip_cost_l786_78641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l786_78621

-- Define the first curve
noncomputable def curve1 (θ : ℝ) : ℝ × ℝ :=
  (Real.sqrt 5 * Real.cos θ, Real.sin θ)

-- Define the second curve
noncomputable def curve2 (t : ℝ) : ℝ × ℝ :=
  (5/4 * t^2, t)

-- Define the domain of θ
def θ_domain (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < Real.pi

-- Theorem statement
theorem intersection_point :
  ∃ (θ t : ℝ), θ_domain θ ∧ curve1 θ = curve2 t ∧ curve2 t = (1, 2 * Real.sqrt 5 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l786_78621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_four_integers_sum_prime_non_existence_of_five_integers_sum_prime_l786_78676

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if the sum of any 3 elements in a set is prime
def sumOfAnyThreeIsPrime (s : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → isPrime (a + b + c)

theorem existence_of_four_integers_sum_prime :
  ∃ s : Finset ℕ, s.card = 4 ∧ sumOfAnyThreeIsPrime s := by
  sorry

theorem non_existence_of_five_integers_sum_prime :
  ¬∃ s : Finset ℕ, s.card = 5 ∧ sumOfAnyThreeIsPrime s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_four_integers_sum_prime_non_existence_of_five_integers_sum_prime_l786_78676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AF_is_sqrt_2_l786_78688

noncomputable def curve (θ : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos θ, 1 + Real.cos (2 * θ))

def F : ℝ × ℝ := (0, 1)

def A : ℝ × ℝ := (1, 0)

theorem distance_AF_is_sqrt_2 :
  let d := Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2)
  d = Real.sqrt 2 := by
  sorry

#check distance_AF_is_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AF_is_sqrt_2_l786_78688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_nested_calc_l786_78644

/-- The ◆ operation for positive real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ := x - 1 / y

/-- Theorem stating the result of the nested diamond operation -/
theorem diamond_nested_calc :
  diamond 3 (diamond (3 + 1) (1 + 2)) = 30 / 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_nested_calc_l786_78644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l786_78683

-- Define the line l: kx - y + 2 = 0
def line (k x y : ℝ) : Prop := k * x - y + 2 = 0

-- Define the circle C: x² + y² - 4x - 12 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 12 = 0

-- Define the intersection points Q and R
def intersection_points (k : ℝ) (Q R : ℝ × ℝ) : Prop :=
  line k Q.1 Q.2 ∧ circle_eq Q.1 Q.2 ∧ line k R.1 R.2 ∧ circle_eq R.1 R.2

-- Define the area of triangle QRC
noncomputable def triangle_area (Q R : ℝ × ℝ) : ℝ :=
  let C : ℝ × ℝ := (2, 0)  -- Center of the circle
  abs ((Q.1 - C.1) * (R.2 - C.2) - (Q.2 - C.2) * (R.1 - C.1)) / 2

-- Theorem statement
theorem max_triangle_area (k : ℝ) (Q R : ℝ × ℝ) :
  intersection_points k Q R →
  ∃ (max_area : ℝ), max_area = 8 ∧ 
    ∀ (Q' R' : ℝ × ℝ), intersection_points k Q' R' → 
      triangle_area Q' R' ≤ max_area :=
by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l786_78683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l786_78645

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x^2 + Real.cos x

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- Define the solution set
def solution_set : Set ℝ := (Set.Ioc (-2) 0) ∪ (Set.Ioo 1 2)

-- Theorem statement
theorem f_inequality_solution (x : ℝ) :
  x ∈ domain ∧ f (2*x - 1) < f 1 ↔ x ∈ solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l786_78645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_on_parabola_l786_78638

noncomputable section

/-- The parabola C: x² = 2y -/
def C (x y : ℝ) : Prop := x^2 = 2*y

/-- The line l: y = 2x + 9/8 -/
def l (x y : ℝ) : Prop := y = 2*x + 9/8

/-- Point A on the parabola C -/
def A : ℝ × ℝ := (-1/2, 1/8)

/-- Point P on the parabola C -/
def P (m : ℝ) : ℝ × ℝ := (m, m^2/2)

/-- Point M: intersection of line through P perpendicular to x-axis and line l -/
def M (m : ℝ) : ℝ × ℝ := (m, 2*m + 9/8)

/-- |AM|² -/
def AM_squared (m : ℝ) : ℝ := 5 * (m + 1/2)^2

/-- |AN| -/
def AN (m : ℝ) : ℝ := Real.sqrt ((m + 1/2)^2 * (1 + 1/4*(m - 1/2)^2 - (4*m - 18)^2/320))

theorem constant_ratio_on_parabola (m : ℝ) 
  (hm1 : m ≠ -1/2) (hm2 : m ≠ 9/2) (hP : C m (m^2/2)) :
  AM_squared m / AN m = 5 * Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_on_parabola_l786_78638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_equality_l786_78679

theorem complex_division_equality : 
  (1 - 3 * Complex.I) / (2 + Complex.I) = -1/5 - 7/5 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_equality_l786_78679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_arithmetic_angles_max_area_of_arithmetic_sides_max_area_conditions_l786_78650

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the circumradius of a triangle
noncomputable def circumradius (t : Triangle) : Real := 
  sorry

-- Define the area of a triangle
noncomputable def area (t : Triangle) : Real := 
  sorry

-- Part 1
theorem circumradius_of_arithmetic_angles (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ d : Real, t.B = t.A + d ∧ t.C = t.B + d)
  (h3 : t.A + t.B + t.C = Real.pi) :
  circumradius t = 2 * Real.sqrt 3 / 3 := by
  sorry

-- Part 2
theorem max_area_of_arithmetic_sides (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ d : Real, t.c = t.b + d ∧ t.a = t.c + d) :
  area t ≤ Real.sqrt 3 := by
  sorry

-- Bonus: Conditions for maximum area
theorem max_area_conditions (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ d : Real, t.c = t.b + d ∧ t.a = t.c + d)
  (h3 : area t = Real.sqrt 3) :
  t.a = 2 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_arithmetic_angles_max_area_of_arithmetic_sides_max_area_conditions_l786_78650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_play_results_in_draw_l786_78660

-- Define the game board
structure GameBoard where
  circles : Finset ℕ
  lines : Set (ℕ × ℕ)

-- Define a player
inductive Player
| White
| Black

-- Define a game state
structure GameState where
  board : GameBoard
  white_pieces : Finset ℕ
  black_pieces : Finset ℕ
  current_player : Player

-- Define an optimal strategy
def OptimalStrategy : GameState → Option ℕ := sorry

-- Define a game result
inductive GameResult
| Win (winner : Player)
| Draw

-- Define the game outcome function
def game_outcome (initial_state : GameState) (white_strategy black_strategy : GameState → Option ℕ) : GameResult := sorry

-- Theorem: If both players play optimally, the game ends in a draw
theorem optimal_play_results_in_draw (initial_state : GameState) :
  game_outcome initial_state OptimalStrategy OptimalStrategy = GameResult.Draw := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_play_results_in_draw_l786_78660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_rate_increase_l786_78651

/-- A bus driver's compensation structure and work details -/
structure BusDriverCompensation where
  regular_rate : ℝ
  regular_hours : ℝ
  total_compensation : ℝ
  total_hours : ℝ

/-- Calculate the percentage increase in overtime pay rate -/
noncomputable def overtime_rate_increase (bdc : BusDriverCompensation) : ℝ :=
  let regular_earnings := bdc.regular_rate * bdc.regular_hours
  let overtime_earnings := bdc.total_compensation - regular_earnings
  let overtime_hours := bdc.total_hours - bdc.regular_hours
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - bdc.regular_rate) / bdc.regular_rate) * 100

/-- Theorem stating the overtime rate increase for the given scenario -/
theorem bus_driver_overtime_rate_increase :
  let bdc : BusDriverCompensation := {
    regular_rate := 12,
    regular_hours := 40,
    total_compensation := 976,
    total_hours := 63.62
  }
  ∃ (ε : ℝ), abs (overtime_rate_increase bdc - 75) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_rate_increase_l786_78651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_number_condition_l786_78674

theorem imaginary_number_condition (m : ℝ) : 
  let z : ℂ := ⟨m * (m - 2), m^2 - 4⟩
  (z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_number_condition_l786_78674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l786_78608

/-- The volume of a pyramid with an isosceles triangular base -/
theorem pyramid_volume (a α β : ℝ) (h_pos : a > 0) (h_α : 0 < α ∧ α < π) (h_β : 0 < β ∧ β < π/2) :
  let base_area : ℝ := (1/2) * a^2 * Real.sin α
  let height : ℝ := (a * Real.tan β) / (2 * Real.cos (α/2))
  let volume : ℝ := (1/3) * base_area * height
  volume = (a^3 * Real.sin (α/2) * Real.tan β) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l786_78608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_2alpha_l786_78602

theorem tan_pi_4_plus_2alpha (α : ℝ) :
  (π < α ∧ α < 3*π/2) →  -- α is in the third quadrant
  Real.cos (2*α) = -3/5 →
  Real.tan (π/4 + 2*α) = -1/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_2alpha_l786_78602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_given_resistance_l786_78622

-- Define the voltage of the battery
noncomputable def voltage : ℝ := 48

-- Define the relationship between current and resistance
noncomputable def current (resistance : ℝ) : ℝ := voltage / resistance

-- State the theorem
theorem current_for_given_resistance :
  current 12 = 4 := by
  -- Unfold the definition of current
  unfold current
  -- Simplify the expression
  simp [voltage]
  -- Evaluate the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_for_given_resistance_l786_78622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_negative_one_l786_78681

theorem tan_alpha_equals_negative_one (α : ℝ) 
  (h1 : Real.sin α - Real.cos α = Real.sqrt 2) 
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_negative_one_l786_78681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_solutions_l786_78666

/-- Predicate defining a valid triangle with given angles and sides -/
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C)

/-- Given a triangle ABC with angle A = 30°, side a = 2, and side b = 2√3, 
    there are exactly 2 possible triangles satisfying these conditions. -/
theorem triangle_two_solutions (A B C : ℝ) (a b c : ℝ) : 
  A = 30 * Real.pi / 180 →  -- Convert 30° to radians
  a = 2 → 
  b = 2 * Real.sqrt 3 → 
  (∃! n : ℕ, n = 2 ∧ 
    ∃ (B₁ C₁ B₂ C₂ : ℝ), 
      (B₁ ≠ B₂ ∨ C₁ ≠ C₂) ∧
      (Triangle A B₁ C₁ a b c ∧ Triangle A B₂ C₂ a b c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_solutions_l786_78666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_theorem_l786_78618

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- Calculates the area of region R -/
noncomputable def area_of_region_R (r : Rhombus) : ℝ :=
  sorry

theorem area_of_region_R_theorem (r : Rhombus) :
  r.side_length = 2 →
  r.angle_B = 2 * Real.pi / 3 →
  area_of_region_R r = 2 * Real.sqrt 3 / 3 := by
  sorry

#check area_of_region_R_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_theorem_l786_78618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l786_78623

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x^2) + Real.sqrt (x^2 - 3)

-- Define the domain of f
def domain : Set ℝ := {x | x = Real.sqrt 3 ∨ x = -Real.sqrt 3}

-- Theorem stating that the range of f is {0}
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l786_78623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AME_l786_78680

-- Define the rectangle ABCD
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = B.2 ∧ C.1 = B.1 ∧ C.2 = D.2 ∧
  B.1 - A.1 = 10 ∧ C.2 - B.2 = 8

-- Define point M as the midpoint of AC
def Midpoint (M A C : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2

-- Define point E on AD with ME perpendicular to AC
def PointE (E A D : ℝ × ℝ) : Prop :=
  E.1 = A.1 ∧ A.2 ≤ E.2 ∧ E.2 ≤ D.2

def Perpendicular (M E A C : ℝ × ℝ) : Prop :=
  (M.1 - E.1) * (C.1 - A.1) + (M.2 - E.2) * (C.2 - A.2) = 0

-- Calculate the area of triangle AME
noncomputable def AreaAME (A M E : ℝ × ℝ) : ℝ :=
  let base := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let height := abs ((E.2 - A.2) * (M.1 - A.1) - (E.1 - A.1) * (M.2 - A.2)) / base
  0.5 * base * height

-- Theorem statement
theorem area_triangle_AME (A B C D M E : ℝ × ℝ) :
  Rectangle A B C D →
  Midpoint M A C →
  PointE E A D →
  Perpendicular M E A C →
  AreaAME A M E = 5 * Real.sqrt 164 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AME_l786_78680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_l786_78670

theorem tan_plus_cot (α : ℝ) (h : Real.sin α - Real.cos α = -Real.sqrt 5 / 2) :
  Real.tan α + (1 / Real.tan α) = -8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_l786_78670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_PF_PF_well_defined_l786_78647

/-- Triangle PQR with altitude PL and median RM -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  F : ℝ × ℝ

/-- The main theorem -/
theorem existence_of_PF (t : Triangle) : ∃ PF : ℝ, PF > 0 :=
  by
  -- Assume the given conditions
  have h1 : t.P.1 = 0 ∧ t.P.2 = 0 := sorry
  have h2 : t.Q.1 = 3 ∧ t.Q.2 = 0 := sorry
  have h3 : t.R.1 = 0 ∧ t.R.2 = 2 * Real.sqrt 3 := sorry
  have h4 : t.M = ((t.P.1 + t.Q.1) / 2, (t.P.2 + t.Q.2) / 2) := sorry
  have h5 : (t.L.1 - t.P.1) * (t.Q.1 - t.R.1) + (t.L.2 - t.P.2) * (t.Q.2 - t.R.2) = 0 := sorry
  have h6 : ∃ k : ℝ, t.F = (k * t.P.1 + (1 - k) * t.L.1, k * t.P.2 + (1 - k) * t.L.2) := sorry
  have h7 : ∃ m : ℝ, t.F = (m * t.R.1 + (1 - m) * t.M.1, m * t.R.2 + (1 - m) * t.M.2) := sorry
  
  -- Prove the existence of PF
  sorry

/-- The distance function -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- PF is well-defined -/
theorem PF_well_defined (t : Triangle) : ∃! PF : ℝ, PF = distance t.P t.F :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_PF_PF_well_defined_l786_78647
