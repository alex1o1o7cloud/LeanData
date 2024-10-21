import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l717_71759

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x)

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (-π/3) 0) := by
  sorry -- Proof is omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l717_71759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_domain_and_range_range_of_m_l717_71730

-- Define the function f(x) = |1 - 1/x|
noncomputable def f (x : ℝ) : ℝ := |1 - 1/x|

-- Part 1: No a and b exist such that domain and range are both [a, b]
theorem no_equal_domain_and_range :
  ¬∃ (a b : ℝ), a < b ∧
  (∀ x, x ∈ Set.Icc a b ↔ f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y) :=
by sorry

-- Part 2: Range of m when domain is [a, b] and range is [ma, mb]
theorem range_of_m (a b m : ℝ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : m ≠ 0) :
  (∀ x, x ∈ Set.Icc a b ↔ f x ∈ Set.Icc (m * a) (m * b)) →
  (∀ y ∈ Set.Icc (m * a) (m * b), ∃ x ∈ Set.Icc a b, f x = y) →
  0 < m ∧ m < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_domain_and_range_range_of_m_l717_71730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_lengths_equal_l717_71702

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a wall section -/
structure WallSection where
  start : Point
  finish : Point

/-- Calculates the length of a wall section -/
noncomputable def wallLength (w : WallSection) : ℝ :=
  Real.sqrt ((w.finish.x - w.start.x)^2 + (w.finish.y - w.start.y)^2)

/-- Theorem: The length of the wall over the hill is equal to the length of the wall on flat ground -/
theorem wall_lengths_equal (A B C : Point) (flatWall hillWall : WallSection) :
  flatWall.start = A ∧ flatWall.finish = B ∧
  hillWall.start = B ∧ hillWall.finish = C ∧
  wallLength flatWall = wallLength { start := A, finish := C } →
  wallLength flatWall = wallLength hillWall := by
  sorry

#check wall_lengths_equal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_lengths_equal_l717_71702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_numbers_l717_71704

def digits : List Nat := [2, 2, 0, 5]

def is_valid_number (n : List Nat) : Bool :=
  n.length = 4 && n.head? ≠ some 0 && n.toFinset ⊆ digits.toFinset

def count_valid_numbers : Nat :=
  (List.permutations digits).filter is_valid_number |>.length

theorem count_four_digit_numbers : count_valid_numbers = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_numbers_l717_71704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l717_71763

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x y : ℝ) : ℝ := (x + y) / (floor x * floor y + floor x + floor y + 1)

theorem f_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  ∃ (z : ℝ), z > 0 ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' * y' = 1 ∧ f x' y' = z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l717_71763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_product_bounds_l717_71771

def numbers : List ℕ := [1, 2, 2, 4, 4, 8, 8]

def is_valid_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 7 ∧ 
  arrangement.toFinset = numbers.toFinset

def product_in_circle (arrangement : List ℕ) (circle : List ℕ) : ℕ :=
  (circle.map (fun i => arrangement.get! i)).prod

def all_circles_equal_product (arrangement : List ℕ) (circles : List (List ℕ)) : Prop :=
  ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → product_in_circle arrangement c1 = product_in_circle arrangement c2

theorem circle_product_bounds (arrangement : List ℕ) (circles : List (List ℕ)) :
  is_valid_arrangement arrangement →
  all_circles_equal_product arrangement circles →
  ∃ P : ℕ, (∀ c, c ∈ circles → product_in_circle arrangement c = P) ∧
           P ≥ 64 ∧ P ≤ 256 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_product_bounds_l717_71771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_problem_l717_71794

/-- Calculates the total amount withdrawn after n years of annual deposits and reinvestments -/
noncomputable def total_amount (a r : ℝ) (n : ℕ) : ℝ :=
  (a / r) * ((1 + r)^(n + 1) - (1 + r))

/-- The problem statement -/
theorem bank_deposit_problem (a r : ℝ) (hr : r > 0) :
  let n : ℕ := 5
  total_amount a r n = (a / r) * ((1 + r)^6 - (1 + r)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_problem_l717_71794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l717_71796

noncomputable def S (n : ℕ) : ℚ := 1 / (3 * n - 2)

noncomputable def a (n : ℕ) : ℚ :=
  if n = 1 then 1
  else S n - S (n - 1)

theorem a_100_value : a 100 = -3/88210 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l717_71796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l717_71783

/-- The slope angle of a line y = kx + 2 passing through (1, 1) is 3π/4 -/
theorem slope_angle_of_line (k : ℝ) : 
  (1 = k * 1 + 2) → 
  Real.arctan k = 3 * Real.pi / 4 := by
  sorry

#check slope_angle_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l717_71783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l717_71726

theorem fraction_sum (a b : ℕ+) (h₁ : Nat.Coprime a.val b.val) 
  (h₂ : (2/3 : ℚ) * a.val^2 / ((1/5 : ℚ) * b.val^2) = 2 * (a.val / b.val)) : 
  a.val + b.val = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_l717_71726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l717_71767

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def isOddOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (-a) a → f (-x) = -f x

def isMonoDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x > f y

-- State the theorem
theorem range_of_m (h_odd : isOddOn f 2) 
                   (h_mono : isMonoDecreasingOn f 0 2) 
                   (h_ineq : ∀ m : ℝ, f (1 - m) < f m) :
  ∀ m : ℝ, m ∈ Set.Icc (-1) (1/2) ↔ 
    (m ∈ Set.Icc (-2) 2 ∧ (1 - m) ∈ Set.Icc (-2) 2 ∧ 1 - m > m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l717_71767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_from_sin_inequality_l717_71711

theorem tan_inequality_from_sin_inequality (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α > Real.sin β) : 
  Real.tan α > Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_from_sin_inequality_l717_71711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l717_71712

/-- Helper function to calculate the distance between two parallel lines -/
noncomputable def distance (l₁ l₂ : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given two lines l₁ and l₂, where l₁ has equal intercepts on x and y axes, 
    prove that the distance between them is √2 -/
theorem distance_between_lines (m : ℝ) (h_m_pos : m > 0) : 
  let l₁ := {(x, y) : ℝ × ℝ | m * x + 2 * y - 4 - m = 0}
  let l₂ := {(x, y) : ℝ × ℝ | x + y - 1 = 0}
  let equal_intercepts := ∃ (a : ℝ), a ≠ 0 ∧ (m * a - 4 - m = 0) ∧ (2 * a - 4 - m = 0)
  equal_intercepts → (distance l₁ l₂ = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l717_71712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sqrt3_3_onto_neg2_0_l717_71787

/-- The projection of vector a onto vector b -/
noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_magnitude_squared := b.1^2 + b.2^2
  let scalar := dot_product / b_magnitude_squared
  (scalar * b.1, scalar * b.2)

/-- Theorem: The projection of (√3, 3) onto (-2, 0) is (√3, 0) -/
theorem projection_sqrt3_3_onto_neg2_0 :
  vector_projection (Real.sqrt 3, 3) (-2, 0) = (Real.sqrt 3, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_sqrt3_3_onto_neg2_0_l717_71787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_line_existence_l717_71745

/-- Two circles with centers O₁ and O₂, radii r and R, intersecting at M -/
structure IntersectingCircles where
  O₁ : EuclideanSpace ℝ (Fin 2)
  O₂ : EuclideanSpace ℝ (Fin 2)
  r : ℝ
  R : ℝ
  M : EuclideanSpace ℝ (Fin 2)
  r_lt_R : r < R
  M_on_circle₁ : dist O₁ M = r
  M_on_circle₂ : dist O₂ M = R

/-- The theorem stating the existence of the required line -/
theorem intersecting_circles_line_existence 
  (circles : IntersectingCircles) (a : ℝ) : 
  ∃ (A B : EuclideanSpace ℝ (Fin 2)), 
    (dist circles.O₁ A = circles.r) ∧ 
    (dist circles.O₂ B = circles.R) ∧
    (dist A B = a) ∧
    (∃ t : ℝ, t ∈ (Set.Ioo 0 1) ∧ circles.M = (1 - t) • A + t • B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_line_existence_l717_71745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_n_l717_71777

/-- Sum of arithmetic progression with n terms, first term a, and common difference d -/
noncomputable def sum_ap (n : ℕ) (a d : ℝ) : ℝ := n / 2 * (2 * a + (n - 1) * d)

/-- R is defined as 2s₃ - 3s₂ + s₁ where sᵢ is the sum of in terms of an AP -/
noncomputable def R (n : ℕ) (a d : ℝ) : ℝ :=
  2 * sum_ap (5 * n) a d - 3 * sum_ap (3 * n) a d + sum_ap n a d

theorem R_depends_on_d_and_n (n : ℕ) (a d : ℝ) :
  R n a d = 5 * n * d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_depends_on_d_and_n_l717_71777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_bike_shop_profit_l717_71756

/-- Represents the profit calculation for Jim's bike shop --/
def bike_shop_profit (tire_repair_price : ℤ) (tire_repair_cost : ℤ) (tire_repairs : ℤ)
                     (complex_repair_price : ℤ) (complex_repair_cost : ℤ) (complex_repairs : ℤ)
                     (retail_profit : ℤ) (fixed_expenses : ℤ) : ℤ :=
  tire_repairs * (tire_repair_price - tire_repair_cost) +
  complex_repairs * (complex_repair_price - complex_repair_cost) +
  retail_profit - fixed_expenses

/-- Theorem stating that Jim's bike shop profit is $3000 --/
theorem jim_bike_shop_profit :
  bike_shop_profit 20 5 300 300 50 2 2000 4000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_bike_shop_profit_l717_71756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_theorem_l717_71780

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_size : ℕ
  num_groups : ℕ
  first_group_start : ℕ
  last_group_end : ℕ

/-- The theorem to be proved -/
theorem systematic_sampling_theorem (s : SystematicSampling)
  (h1 : s.total_students = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.group_size = 8)
  (h4 : s.num_groups = 20)
  (h5 : s.first_group_start = 1)
  (h6 : s.last_group_end = 160)
  (h7 : ∀ i : ℕ, i ≤ s.num_groups → 
    (i - 1) * s.group_size + 1 ≤ (i * s.group_size))
  (h8 : 126 ∈ Finset.Icc ((16 - 1) * s.group_size + 1) (16 * s.group_size)) :
  6 ∈ Finset.Icc s.first_group_start (s.first_group_start + s.group_size - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_theorem_l717_71780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l717_71707

/-- The volume of a specific part of a frustum given its dimensions -/
theorem frustum_volume (h r₁ r₂ : ℝ) (h_pos : h > 0) (r₁_pos : r₁ > 0) (r₂_pos : r₂ > 0) :
  (π * h / 3) * (r₁ * r₂ * (r₁^2 + 3*r₁*r₂ + r₂^2)) / ((r₁ + r₂)^2) =
  (π * h / 3) * (r₁ * r₂ * (r₁^2 + 3*r₁*r₂ + r₂^2)) / ((r₁ + r₂)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_l717_71707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_ties_hockey_game_l717_71732

noncomputable def binomialCoeff (n k : ℕ) : ℚ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def tieProb (k : ℕ) : ℚ :=
  binomialCoeff (2 * k) k / (4 ^ k)

noncomputable def expectedTies (n : ℕ) : ℚ :=
  List.range n |>.map (fun k => tieProb (k + 1)) |>.sum

theorem expected_ties_hockey_game :
  ∃ (ε : ℚ), abs (expectedTies 5 - 1707/1000) < ε ∧ ε < 1/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_ties_hockey_game_l717_71732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_inequalities_l717_71725

theorem two_correct_inequalities (a b : ℝ) (h : a < b ∧ b < 0) :
  (ite (a + b < a * b) 1 0) + (ite (abs a > abs b) 1 0) + 
  (ite (a < b) 1 0) + (ite (a^2 + b^2 > 2) 1 0) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_correct_inequalities_l717_71725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_and_angles_l717_71748

/-- The plane passing through points M(1, 2, 0), N(1, -1, 2), and P(0, 1, -1) -/
def plane_through_MNP (x y z : ℝ) : Prop :=
  5 * x - 2 * y + 3 * z - 1 = 0

/-- The points M, N, and P -/
def M : ℝ × ℝ × ℝ := (1, 2, 0)
def N : ℝ × ℝ × ℝ := (1, -1, 2)
def P : ℝ × ℝ × ℝ := (0, 1, -1)

/-- The normal vector of the plane -/
def normal_vector : ℝ × ℝ × ℝ := (5, -2, 3)

theorem plane_equation_and_angles :
  (plane_through_MNP M.1 M.2.1 M.2.2 ∧
   plane_through_MNP N.1 N.2.1 N.2.2 ∧
   plane_through_MNP P.1 P.2.1 P.2.2) ∧
  (normal_vector.1 / Real.sqrt 38 = 5 / Real.sqrt 38 ∧
   normal_vector.2.1 / Real.sqrt 38 = -2 / Real.sqrt 38 ∧
   normal_vector.2.2 / Real.sqrt 38 = 3 / Real.sqrt 38) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_and_angles_l717_71748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_recipe_correct_l717_71701

/-- Represents a cookie recipe -/
structure Recipe where
  cookies : ℕ
  flour : ℚ
  sugar : ℚ

/-- Calculates the scaled recipe with reduced sugar -/
def scale_recipe (r : Recipe) (new_cookies : ℕ) (sugar_reduction : ℚ) : Recipe :=
  let scale_factor : ℚ := (new_cookies : ℚ) / (r.cookies : ℚ)
  { cookies := new_cookies,
    flour := r.flour * scale_factor,
    sugar := r.sugar * scale_factor * (1 - sugar_reduction) }

/-- The original recipe -/
def original_recipe : Recipe :=
  { cookies := 40, flour := 3, sugar := 1 }

/-- The theorem stating the correct amounts for the scaled recipe -/
theorem scaled_recipe_correct :
  let scaled := scale_recipe original_recipe 80 (1/4)
  scaled.flour = 6 ∧ scaled.sugar = (3/2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_recipe_correct_l717_71701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l717_71700

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x

def f' (x : ℝ) : ℝ := Real.log x + 1

theorem tangent_line_and_monotonicity :
  (∀ x : ℝ, x > 0 → deriv f x = Real.log x + 1) →
  (∀ x y : ℝ, y = 2*x - Real.exp 1 ↔ y - f (Real.exp 1) = f' (Real.exp 1) * (x - Real.exp 1)) ∧
  (∀ x : ℝ, 0 < x → x < 1/(Real.exp 1) → (f' x < 0)) ∧
  (∀ x : ℝ, x > 1/(Real.exp 1) → (f' x > 0)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l717_71700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l717_71788

-- Define the slopes of two lines
noncomputable def m1 (a : ℝ) : ℝ := -a / (2*a - 1)
noncomputable def m2 (a : ℝ) : ℝ := -3 / a

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := m1 a * m2 a = -1

-- Theorem statement
theorem perpendicular_lines (a : ℝ) :
  perpendicular a → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l717_71788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dallas_zoo_birds_l717_71773

theorem dallas_zoo_birds (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 300) (h2 : total_legs = 680) : ∃ b m i : ℕ,
  b + m + i = total_heads ∧
  2*b + 4*m + 6*i = total_legs ∧
  b = 280 := by
  -- Define variables
  let birds := λ (b m i : ℕ) => b
  let mammals := λ (b m i : ℕ) => m
  let insects := λ (b m i : ℕ) => i
  let total_animals := λ (b m i : ℕ) => b + m + i
  let total_legs_count := λ (b m i : ℕ) => 2*b + 4*m + 6*i

  -- Proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dallas_zoo_birds_l717_71773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_N_l717_71727

noncomputable def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}
noncomputable def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_N_l717_71727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l717_71755

def is_valid_subset (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, ∀ y ∈ S, x ≤ 100 ∧ y ≤ 100 ∧ (3 * x ≠ y) ∧ (3 * y ≠ x)

theorem max_subset_size :
  ∃ (S : Finset ℕ), is_valid_subset S ∧ S.card = 76 ∧
  ∀ (T : Finset ℕ), is_valid_subset T → T.card ≤ 76 := by
  sorry

#check max_subset_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l717_71755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_conical_container_l717_71717

/-- The height of water in a conical container with equilateral triangle cross-section,
    given that it originally filled a cylinder of radius 2cm and height 6cm. -/
theorem water_height_in_conical_container (r h : ℝ) : 
  r = 2 → h = 6 → ∃ (h_cone : ℝ), h_cone = 6 ∧ 
  (1 / 3 : ℝ) * Real.pi * (2 * Real.sqrt 3)^2 * h_cone = Real.pi * r^2 * h := by
  sorry

#check water_height_in_conical_container

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_conical_container_l717_71717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_approx_l717_71729

/-- The radius of a cone's base given its slant height and curved surface area -/
noncomputable def cone_base_radius (slant_height : ℝ) (curved_surface_area : ℝ) : ℝ :=
  curved_surface_area / (Real.pi * slant_height)

/-- Theorem: The radius of a cone's base is approximately 12 cm given the specified conditions -/
theorem cone_base_radius_approx :
  let slant_height : ℝ := 14
  let curved_surface_area : ℝ := 527.7875658030853
  abs (cone_base_radius slant_height curved_surface_area - 12) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_approx_l717_71729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_sum_l717_71741

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the condition given in the problem
def TriangleCondition (A B C : ℝ) (a b c : ℝ) : Prop :=
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C

-- State the theorem
theorem max_sin_sum (A B C : ℝ) (a b c : ℝ) 
  (h1 : Triangle A B C a b c) 
  (h2 : TriangleCondition A B C a b c) : 
  (∀ A' B' C' a' b' c', Triangle A' B' C' a' b' c' → 
    TriangleCondition A' B' C' a' b' c' → 
    Real.sin A' + Real.sin B' ≤ Real.sqrt 3) ∧ 
  (∃ A' B' C' a' b' c', Triangle A' B' C' a' b' c' ∧ 
    TriangleCondition A' B' C' a' b' c' ∧ 
    Real.sin A' + Real.sin B' = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_sum_l717_71741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_locus_is_circle_l717_71724

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a ray in 3D space -/
structure Ray3D where
  origin : Point3D
  direction : Point3D

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Represents a circle in 3D space -/
structure Circle3D where
  center : Point3D
  radius : ℝ

/-- The incenter of a triangle -/
noncomputable def incenter (t : Triangle3D) : Point3D :=
  sorry

/-- Check if a triangle is isosceles -/
def isIsosceles (t : Triangle3D) : Prop :=
  sorry

/-- Check if three planar angles of a trihedral angle are equal -/
def hasEqualPlanarAngles (A B C X : Point3D) : Prop :=
  sorry

/-- The locus of a point -/
def locus (f : ℝ → Point3D) : Set Point3D :=
  sorry

/-- Point on a ray at a given parameter t -/
def Ray3D.pointAt (r : Ray3D) (t : ℝ) : Point3D :=
  sorry

/-- Convert a Circle3D to a Set Point3D -/
def Circle3D.toSet (c : Circle3D) : Set Point3D :=
  sorry

/-- Main theorem -/
theorem incenter_locus_is_circle 
  (ABC : Triangle3D) 
  (x : Ray3D) 
  (h1 : isIsosceles ABC)
  (h2 : hasEqualPlanarAngles ABC.A ABC.B ABC.C x.direction) :
  ∃ (c : Circle3D), locus (λ t ↦ incenter (Triangle3D.mk (x.pointAt t) ABC.B ABC.C)) ⊆ c.toSet :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_locus_is_circle_l717_71724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_radius_and_sector_area_l717_71739

/-- Given an arc length and central angle, prove the radius and sector area -/
theorem arc_radius_and_sector_area 
  (arc_length : ℝ) 
  (central_angle : ℝ) 
  (h_arc_length : arc_length = 50)
  (h_central_angle : central_angle = 200) :
  ∃ (radius : ℝ) (sector_area : ℝ),
    (radius = 45 / Real.pi) ∧
    (sector_area = 1125 / Real.pi) ∧
    (arc_length = central_angle * (Real.pi / 180) * radius) ∧ 
    (sector_area = (1 / 2) * arc_length * radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_radius_and_sector_area_l717_71739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l717_71765

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

theorem f_properties :
  (∀ x, x > 0 → f 0 x ≤ -1) ∧
  (f 0 1 = -1) ∧
  (∀ a : ℝ, (∃! x, x > 0 ∧ f a x = 0) ↔ a > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l717_71765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l717_71779

theorem smallest_value_of_expression (a b : ℤ) (ha : 0 < a ∧ a < 6) (hb : 0 < b ∧ b < 10) :
  (∀ x y : ℤ, 0 < x ∧ x < 6 → 0 < y ∧ y < 10 → 2 * x - x * y ≥ 2 * a - a * b) →
  2 * a - a * b = -35 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_of_expression_l717_71779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_124th_term_l717_71792

def sequenceZ (a b : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | (n+2) => sequenceZ a b (n+1) - sequenceZ a b n

theorem sequence_124th_term (a b : ℤ) : sequenceZ a b 123 = -a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_124th_term_l717_71792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_inequality_l717_71760

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.00010101 * (10 : ℝ)^m ≤ 1000)) ∧ 
  (0.00010101 * (10 : ℝ)^k > 1000) → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_for_inequality_l717_71760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_diameter_l717_71775

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The radius of a sphere given its volume -/
noncomputable def sphere_radius (v : ℝ) : ℝ := ((3 * v) / (4 * Real.pi))^(1/3)

theorem larger_sphere_diameter (r : ℝ) (h : r = 6) :
  2 * sphere_radius (3 * sphere_volume r) = 12 * Real.rpow 6 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_diameter_l717_71775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_for_given_canoe_speeds_l717_71761

/-- The speed of a stream given upstream and downstream canoe speeds -/
noncomputable def stream_speed (upstream_speed downstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem stream_speed_for_given_canoe_speeds :
  stream_speed 6 10 = 2 := by
  -- Unfold the definition of stream_speed
  unfold stream_speed
  -- Simplify the arithmetic
  simp [sub_div]
  -- The result follows from real number arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_for_given_canoe_speeds_l717_71761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l717_71764

noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (4/5) * t, -1 - (3/5) * t)

noncomputable def curve (θ : ℝ) : ℝ := Real.sqrt 2 * Real.cos (θ + Real.pi/4)

theorem chord_length :
  ∃ (t₁ t₂ : ℝ) (θ₁ θ₂ : ℝ),
    let (x₁, y₁) := line t₁
    let (x₂, y₂) := line t₂
    let ρ₁ := curve θ₁
    let ρ₂ := curve θ₂
    x₁ = ρ₁ * Real.cos θ₁ ∧
    y₁ = ρ₁ * Real.sin θ₁ ∧
    x₂ = ρ₂ * Real.cos θ₂ ∧
    y₂ = ρ₂ * Real.sin θ₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 7/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l717_71764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_l717_71705

/-- The distance between home and school -/
noncomputable def distance : ℝ := 2.5

/-- The time it takes to travel when arriving exactly on time (in hours) -/
noncomputable def on_time : ℝ := 5 / 12

theorem school_distance :
  (∀ (v₁ v₂ : ℝ),
    v₁ = 5 ∧ v₂ = 10 →
    v₁ * (on_time + 5 / 60) = distance ∧
    v₂ * (on_time - 10 / 60) = distance) →
  distance = 2.5 := by
  sorry

#check school_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_l717_71705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_l717_71703

theorem distance_product (b₁ b₂ : ℝ) : 
  (∀ b ∈ ({b₁, b₂} : Set ℝ), ((3*b - 7)^2 + (2*b - 8)^2 : ℝ) = (3*Real.sqrt 13)^2) →
  b₁ * b₂ = -1/169 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_l717_71703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_construction_possible_l717_71746

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a regular convex dodecagon -/
structure Dodecagon where
  vertices : Fin 12 → Point

/-- Represents a compass -/
structure Compass where
  center : Point
  radius : ℝ

/-- Check if a point is on a circle drawn by a compass -/
def Point.isOnCircle (p : Point) (c : Compass) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Given two vertices of a dodecagon, construct all vertices using only a compass -/
noncomputable def construct_dodecagon (A₁ A₆ : Point) : Dodecagon :=
  sorry

/-- The theorem stating that it's possible to construct a dodecagon given two vertices -/
theorem dodecagon_construction_possible (A₁ A₆ : Point) :
  ∃ (d : Dodecagon), d.vertices 0 = A₁ ∧ d.vertices 5 = A₆ ∧
  (∀ i : Fin 12, ∃ (c : Compass), d.vertices i = c.center ∨ (d.vertices i).isOnCircle c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_construction_possible_l717_71746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_given_monomial_l717_71723

/-- The coefficient of a monomial is the numerical factor multiplying the variable terms. -/
noncomputable def coefficient (monomial : ℝ → ℝ → ℝ) : ℝ := sorry

/-- Given monomial -/
noncomputable def given_monomial (m n : ℝ) : ℝ := -((2 * Real.pi) / 3) * m * (n^5)

/-- Theorem: The coefficient of the monomial -2π/3 * m * n^5 is -2π/3 -/
theorem coefficient_of_given_monomial :
  coefficient given_monomial = -(2 * Real.pi / 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_given_monomial_l717_71723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_siblings_is_four_fifths_l717_71744

/-- Represents a group of people in a room with sibling relationships -/
structure SiblingGroup where
  people : Finset ℕ
  siblings : ℕ → Finset ℕ
  h_size : people.card = 6
  h_siblings : ∀ p, p ∈ people → (siblings p).card = 2 ∧ siblings p ⊆ people
  h_symmetry : ∀ p q, p ∈ people → q ∈ people → (q ∈ siblings p ↔ p ∈ siblings q)

/-- The probability of selecting two non-siblings from a SiblingGroup -/
def prob_non_siblings (g : SiblingGroup) : ℚ :=
  let total_pairs := Nat.choose g.people.card 2
  let sibling_pairs := (g.people.filter (λ p => p ∈ g.siblings p)).card
  (total_pairs - sibling_pairs : ℚ) / total_pairs

/-- Theorem stating the probability of selecting two non-siblings is 4/5 -/
theorem prob_non_siblings_is_four_fifths (g : SiblingGroup) :
  prob_non_siblings g = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_siblings_is_four_fifths_l717_71744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_iff_tangent_circles_l717_71793

/-- Represents a circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate for two points being on the common external tangent of two circles -/
def ExternalTangent (Γ₁ Γ₂ : Circle) (P Q : Point) : Prop :=
  sorry

/-- Predicate for four points forming a cyclic quadrilateral -/
def Cyclic (A B C D : Point) : Prop :=
  sorry

/-- Predicate for two circles being externally tangent -/
def ExternallyTangent (Γ₁ Γ₂ : Circle) : Prop :=
  sorry

/-- Two circles with different radii and common external tangents form a cyclic quadrilateral if and only if the circles are externally tangent -/
theorem cyclic_quadrilateral_iff_tangent_circles (Γ₁ Γ₂ : Circle) (A B C D : Point) :
  Γ₁.radius ≠ Γ₂.radius →
  ExternalTangent Γ₁ Γ₂ A B →
  ExternalTangent Γ₁ Γ₂ C D →
  Cyclic A B C D ↔ ExternallyTangent Γ₁ Γ₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_iff_tangent_circles_l717_71793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l717_71790

theorem sum_of_roots_quadratic : 
  ∃ (x₁ x₂ : ℝ), x₁^2 - 7*x₁ + 10 = 0 ∧ x₂^2 - 7*x₂ + 10 = 0 ∧ x₁ + x₂ = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_l717_71790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l717_71733

theorem negation_equivalence (a b : ℝ) :
  ¬(a > b → (2 : ℝ)^a > (2 : ℝ)^b) ↔ (a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l717_71733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l717_71736

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ side : ℝ, new_area = (4 * side)^2 * (Real.sqrt 3 / 4)) →
  new_area = 144 →
  original_area = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l717_71736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l717_71731

/-- The equation defining the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 8*y = 16

/-- The area of the region defined by the equation -/
noncomputable def region_area : ℝ := 41 * Real.pi

/-- Theorem stating that the region is a circle and its area is correct -/
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l717_71731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_triangle_prob_value_l717_71774

/-- The number of edges in a complete graph with 6 vertices -/
def num_edges : ℕ := 15

/-- The number of possible triangles in a complete graph with 6 vertices -/
def num_triangles : ℕ := 20

/-- The probability that a single edge is colored with a specific color -/
def single_color_prob : ℚ := 1 / 3

/-- The probability that a given triangle has all sides of different colors -/
def diff_color_triangle_prob : ℚ := 2 / 9

/-- The probability that a given triangle does not have all sides of different colors -/
def not_diff_color_triangle_prob : ℚ := 1 - diff_color_triangle_prob

/-- The theorem stating the probability of the existence of a triangle with all sides of different colors -/
def hexagon_triangle_prob (p : ℚ) : Prop :=
  p = 1 - not_diff_color_triangle_prob ^ num_triangles

/-- The main theorem to be proved -/
theorem hexagon_triangle_prob_value : 
  hexagon_triangle_prob (1 - (7/9)^20) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_triangle_prob_value_l717_71774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_problem_l717_71766

/-- Represents a quadratic polynomial ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic polynomial at a given x -/
noncomputable def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Represents a rational function p(x)/q(x) where p and q are quadratic polynomials -/
structure RationalFunction where
  p : QuadraticPolynomial
  q : QuadraticPolynomial

/-- Evaluates a rational function at a given x -/
noncomputable def RationalFunction.eval (f : RationalFunction) (x : ℝ) : ℝ :=
  (f.p.eval x) / (f.q.eval x)

/-- Checks if a rational function has a vertical asymptote at a given x -/
def has_vertical_asymptote (f : RationalFunction) (x : ℝ) : Prop :=
  f.q.eval x = 0 ∧ f.p.eval x ≠ 0

/-- Checks if a rational function has a horizontal asymptote at a given y -/
def has_horizontal_asymptote (f : RationalFunction) (y : ℝ) : Prop :=
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f.eval x - y| < ε

/-- Checks if a rational function has a hole at a given x -/
def has_hole (f : RationalFunction) (x : ℝ) : Prop :=
  f.p.eval x = 0 ∧ f.q.eval x = 0

theorem rational_function_problem (f : RationalFunction) :
  has_vertical_asymptote f (-3) →
  has_horizontal_asymptote f (-4) →
  has_hole f 4 →
  f.eval 2 = 0 →
  f.eval 2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_problem_l717_71766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_third_l717_71795

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 17 * x - 6) / (x - 1/3)

theorem limit_of_f_at_one_third :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 1/3 →
    |x - 1/3| < δ → |f x - 19| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_one_third_l717_71795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicularity_condition_l717_71709

/-- Two lines are perpendicular if and only if the sum of the products of their coefficients is zero -/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The first line equation: (m+2)x+(1-m)y=0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 2) * x + (1 - m) * y = 0

/-- The second line equation: (m-1)x+(2m+3)y+2=0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x + (2 * m + 3) * y + 2 = 0

/-- The perpendicularity condition for the given lines -/
def lines_are_perpendicular (m : ℝ) : Prop :=
  are_perpendicular (m + 2) (1 - m) (m - 1) (2 * m + 3)

/-- The statement to be proved -/
theorem perpendicularity_condition :
  ¬(∀ m : ℝ, m = 1 → lines_are_perpendicular m) ∧
  ¬(∀ m : ℝ, lines_are_perpendicular m → m = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicularity_condition_l717_71709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unhappiness_theorem_l717_71781

/-- The number of students in the class -/
def num_students : ℕ := 2017

/-- The number of groups the class should be split into -/
def num_groups : ℕ := 15

/-- The unhappiness levels of the students -/
def unhappiness_levels : List ℕ := List.range' 1 num_students

/-- The unhappiness level of a group is the average unhappiness of its members -/
def group_unhappiness (group : List ℕ) : ℚ :=
  (group.sum : ℚ) / group.length

/-- The unhappiness of the class is the sum of the unhappiness of all groups -/
def class_unhappiness (groups : List (List ℕ)) : ℚ :=
  (groups.map group_unhappiness).sum

/-- The theorem stating the minimum unhappiness of the class -/
theorem min_unhappiness_theorem :
  ∃ (partition : List (List ℕ)),
    partition.length = num_groups ∧
    (partition.join.toFinset = unhappiness_levels.toFinset) ∧
    ∀ (other_partition : List (List ℕ)),
      other_partition.length = num_groups →
      (other_partition.join.toFinset = unhappiness_levels.toFinset) →
      class_unhappiness partition ≤ class_unhappiness other_partition ∧
      class_unhappiness partition = 1943.2667 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unhappiness_theorem_l717_71781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_lines_l717_71772

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- Define the integral bounds
noncomputable def a : ℝ := 2 * Real.pi / 3
noncomputable def b : ℝ := Real.pi

-- Define the area S
noncomputable def S : ℝ := -(∫ x in a..b, f x)

-- Theorem statement
theorem area_enclosed_by_curve_and_lines : S = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_and_lines_l717_71772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_similar_statues_l717_71797

/-- Calculates the paint required for multiple similar statues -/
theorem paint_for_similar_statues 
  (original_height : ℝ) 
  (original_paint : ℝ) 
  (new_height : ℝ) 
  (num_statues : ℕ) 
  (h1 : original_height > 0) 
  (h2 : original_paint > 0) 
  (h3 : new_height > 0) :
  let volume_ratio := (new_height / original_height) ^ 3
  let paint_per_statue := original_paint * volume_ratio
  let total_paint := paint_per_statue * (num_statues : ℝ)
  (original_height = 6 ∧ 
   original_paint = 1 ∧ 
   new_height = 2 ∧ 
   num_statues = 1000) → 
  ⌊total_paint⌋ = 37 := by
  sorry

#check paint_for_similar_statues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_similar_statues_l717_71797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l717_71789

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (k₁ k₂ : Circle) (T M₁ M₂ P : ℝ × ℝ) : Prop :=
  -- Circles have different radii
  k₁.radius ≠ k₂.radius ∧
  -- Circles touch at point T
  Real.sqrt ((k₁.center.1 - T.1)^2 + (k₁.center.2 - T.2)^2) = k₁.radius ∧
  Real.sqrt ((k₂.center.1 - T.1)^2 + (k₂.center.2 - T.2)^2) = k₂.radius ∧
  -- M₁ and M₂ are on the circles
  Real.sqrt ((k₁.center.1 - M₁.1)^2 + (k₁.center.2 - M₁.2)^2) = k₁.radius ∧
  Real.sqrt ((k₂.center.1 - M₂.1)^2 + (k₂.center.2 - M₂.2)^2) = k₂.radius ∧
  -- T, M₁, and M₂ are collinear
  ∃ t : ℝ, M₁ = (1 - t) • T + t • M₂ ∧
  -- P is the intersection of O₁M₂ and O₂M₁
  ∃ s₁ s₂ : ℝ, 
    P = (1 - s₁) • k₁.center + s₁ • M₂ ∧
    P = (1 - s₂) • k₂.center + s₂ • M₁

-- Define the locus of P
def locus_of_P (k₁ k₂ : Circle) (T : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ M₁ M₂, problem_setup k₁ k₂ T M₁ M₂ P}

-- Theorem statement
theorem locus_is_circle (k₁ k₂ : Circle) (T : ℝ × ℝ) :
  ∃ C : Circle, locus_of_P k₁ k₂ T = {P | Real.sqrt ((C.center.1 - P.1)^2 + (C.center.2 - P.2)^2) = C.radius} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l717_71789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_hill_time_l717_71753

/-- Calculates the total time for a cyclist to climb and descend a hill. -/
noncomputable def total_time (hill_length : ℝ) (climbing_speed : ℝ) : ℝ :=
  let descending_speed := 2 * climbing_speed
  let time_to_climb := hill_length / climbing_speed
  let time_to_descend := hill_length / descending_speed
  time_to_climb + time_to_descend

/-- Proves that the total time for a cyclist to climb and descend a hill is 300 seconds. -/
theorem cyclist_hill_time :
  let hill_length : ℝ := 400
  let climbing_speed : ℝ := 2
  total_time hill_length climbing_speed = 300 := by
  -- Unfold the definition of total_time
  unfold total_time
  -- Simplify the expressions
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_hill_time_l717_71753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_zero_l717_71776

theorem trig_sum_zero : 
  Real.sin (-π/3) + 2 * Real.sin (4*π/3) + 3 * Real.sin (2*π/3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_zero_l717_71776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_area_ratio_l717_71786

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Given a line y = b + x, prove that if the ratio of areas of triangles QRS and QOP is 4:16,
    then b = -4/3, where Q, R, S, O, P are as defined in the problem statement. -/
theorem line_intersection_area_ratio (b : ℝ) : 
  (∃ (Q R S O P : ℝ × ℝ),
    Q.1 = -b ∧ Q.2 = 0 ∧ 
    R = (4, 0) ∧
    S = (4, b + 4) ∧
    O = (0, 0) ∧
    P = (0, b) ∧
    (area_triangle Q R S) / (area_triangle Q O P) = 4 / 16) →
  b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_area_ratio_l717_71786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_is_three_eighths_l717_71754

/-- Represents the contents of a cup -/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Represents the state of both cups -/
structure TwoGlasses where
  cup1 : CupContents
  cup2 : CupContents

/-- Initial setup of the cups -/
def initial_state : TwoGlasses :=
  { cup1 := { tea := 6, milk := 0 },
    cup2 := { tea := 0, milk := 6 } }

/-- Pour from cup1 to cup2 -/
def pour_1_to_2 (state : TwoGlasses) (fraction : ℚ) : TwoGlasses :=
  let amount := fraction * state.cup1.tea
  { cup1 := { tea := state.cup1.tea - amount, milk := state.cup1.milk },
    cup2 := { tea := state.cup2.tea + amount, milk := state.cup2.milk } }

/-- Pour from cup2 to cup1 -/
def pour_2_to_1 (state : TwoGlasses) (fraction : ℚ) : TwoGlasses :=
  let total2 := state.cup2.tea + state.cup2.milk
  let amount := fraction * total2
  let tea_fraction := state.cup2.tea / total2
  let milk_fraction := state.cup2.milk / total2
  { cup1 := { tea := state.cup1.tea + amount * tea_fraction,
              milk := state.cup1.milk + amount * milk_fraction },
    cup2 := { tea := state.cup2.tea - amount * tea_fraction,
              milk := state.cup2.milk - amount * milk_fraction } }

/-- Perform the series of pourings described in the problem -/
def final_state : TwoGlasses :=
  let state1 := pour_1_to_2 initial_state (1/3)
  let state2 := pour_2_to_1 state1 (1/2)
  pour_1_to_2 state2 (1/4)

/-- The fraction of milk in the first cup after the pourings -/
def milk_fraction (state : TwoGlasses) : ℚ :=
  let total1 := state.cup1.tea + state.cup1.milk
  state.cup1.milk / total1

theorem milk_fraction_is_three_eighths :
  milk_fraction final_state = 3/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_is_three_eighths_l717_71754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l717_71791

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_decreasing_neg : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f y ≤ f x

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo 0 (1/100) ∪ Set.Ioi 100

-- Theorem statement
theorem inequality_solution : 
  {x : ℝ | f (Real.log x) > f (-2)} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l717_71791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l717_71720

noncomputable def complex_sum : ℂ :=
  Complex.exp (3 * Real.pi * Complex.I / 60) +
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (19 * Real.pi * Complex.I / 60) +
  Complex.exp (27 * Real.pi * Complex.I / 60) +
  Complex.exp (35 * Real.pi * Complex.I / 60) +
  Complex.exp (43 * Real.pi * Complex.I / 60) +
  Complex.exp (51 * Real.pi * Complex.I / 60) +
  Complex.exp (59 * Real.pi * Complex.I / 60)

theorem complex_sum_argument :
  Complex.arg complex_sum = 31 * Real.pi / 60 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_argument_l717_71720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l717_71770

/-- A function g defined on a domain of three points -/
def g (x : ℝ) : ℝ := sorry

/-- The area of the triangle formed by the points on the graph of y = g(x) -/
def area_g : ℝ := 50

/-- The area of the triangle formed by the points on the graph of y = 3g(3x) -/
noncomputable def area_3g3x : ℝ := 
  let x_a : ℝ := sorry
  let x_b : ℝ := sorry
  let x_c : ℝ := sorry
  let points_g := [(x_a, g x_a), (x_b, g x_b), (x_c, g x_c)]
  let points_3g3x := [(x_a/3, 3*(g x_a)), (x_b/3, 3*(g x_b)), (x_c/3, 3*(g x_c))]
  sorry -- Placeholder for area calculation

/-- The theorem stating that the areas are equal -/
theorem area_equality : area_g = area_3g3x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l717_71770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l717_71740

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point F on BC extended
noncomputable def F (A B C : ℝ × ℝ) : ℝ × ℝ := (2/3 : ℝ) • C + (1/3 : ℝ) • B

-- Define point G on AC
noncomputable def G (A B C : ℝ × ℝ) : ℝ × ℝ := (1/5 : ℝ) • A + (4/5 : ℝ) • C

-- Define the intersection point Q
noncomputable def Q (A B C : ℝ × ℝ) : ℝ × ℝ := (5/8 : ℝ) • A + (3/8 : ℝ) • B + (1/2 : ℝ) • C

-- Theorem statement
theorem intersection_point_coordinates (A B C : ℝ × ℝ) :
  let F := F A B C
  let G := G A B C
  let Q := Q A B C
  (∃ t : ℝ, Q = (1 - t) • A + t • F) ∧ (∃ s : ℝ, Q = (1 - s) • B + s • G) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l717_71740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l717_71719

/-- Payment calculation for Harry -/
noncomputable def harry_payment (x : ℝ) (h : ℝ) : ℝ :=
  if h ≤ 12 then x * h else x * 12 + 1.5 * x * (h - 12)

/-- Payment calculation for James -/
noncomputable def james_payment (x : ℝ) (h : ℝ) : ℝ :=
  if h ≤ 40 then x * h else x * 40 + 2 * x * (h - 40)

theorem harry_hours_worked (x : ℝ) (h : ℝ) :
  x > 0 →
  james_payment x 41 = harry_payment x h →
  h = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l717_71719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_calories_burned_l717_71758

/-- Calculates the number of calories burned per week in spinning classes -/
def calories_burned_per_week (classes_per_week : ℕ) (hours_per_class : ℚ) (calories_per_minute : ℕ) : ℕ :=
  let minutes_per_class : ℕ := (hours_per_class * 60).floor.toNat
  let calories_per_class : ℕ := minutes_per_class * calories_per_minute
  classes_per_week * calories_per_class

/-- Proves that James burns 1890 calories per week from his spinning classes -/
theorem james_calories_burned :
  calories_burned_per_week 3 (3/2) 7 = 1890 := by
  rfl

#eval calories_burned_per_week 3 (3/2) 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_calories_burned_l717_71758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_odd_l717_71757

noncomputable def f (x : ℝ) := 2 * Real.sin (x / 2 - Real.pi / 6)

noncomputable def g (x : ℝ) := f (x + Real.pi / 3)

theorem f_shifted_is_odd : Odd g := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_is_odd_l717_71757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_regular_octagon_l717_71706

/-- A regular octagon inscribed in a circle -/
structure RegularOctagon where
  /-- The length of each side of the octagon -/
  side_length : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length

/-- The length of the arc intercepted by one side of the octagon -/
noncomputable def arc_length (octagon : RegularOctagon) : ℝ :=
  (5 * Real.pi) / 4

/-- Theorem: The length of the arc intercepted by one side of a regular octagon
    with side length 5 is (5π)/4 -/
theorem arc_length_of_regular_octagon (octagon : RegularOctagon)
    (h : octagon.side_length = 5) :
    arc_length octagon = (5 * Real.pi) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_regular_octagon_l717_71706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l717_71737

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f a + f (2 * b - 1) = 0) :
  (1 / a + 4 / b) ≥ 9 + 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l717_71737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pancake_breakfast_theorem_l717_71713

def pancake_breakfast (pancake_cost bacon_cost : ℚ) (pancakes_sold : ℕ) (total_raised : ℚ) : ℕ :=
  let bacon_revenue := total_raised - (pancake_cost * pancakes_sold)
  (bacon_revenue / bacon_cost).floor.toNat

theorem pancake_breakfast_theorem (pancake_cost bacon_cost : ℚ) (pancakes_sold : ℕ) (total_raised : ℚ) :
  pancake_cost = 4 →
  bacon_cost = 2 →
  pancakes_sold = 60 →
  total_raised = 420 →
  pancake_breakfast pancake_cost bacon_cost pancakes_sold total_raised = 90 :=
by
  sorry

#eval pancake_breakfast 4 2 60 420

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pancake_breakfast_theorem_l717_71713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_threeDigitMultiplesOf3Count_l717_71716

-- Define the set of digits
def digits : Finset ℕ := Finset.range 10

-- Define a function to check if a number is divisible by 3
def divisibleBy3 (n : ℕ) : Bool := n % 3 = 0

-- Define a function to create a three-digit number from three digits
def makeThreeDigitNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Main theorem
theorem threeDigitMultiplesOf3Count : 
  (Finset.filter (λ n => divisibleBy3 n ∧ n ≥ 100 ∧ n < 1000) 
    (Finset.image (λ (abc : ℕ × ℕ × ℕ) => makeThreeDigitNumber abc.1 abc.2.1 abc.2.2) 
      (digits.product (digits.product digits)))).card = 228 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_threeDigitMultiplesOf3Count_l717_71716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l717_71734

theorem smallest_n_for_sqrt_difference (n : ℕ) : 
  (n ≥ 626 ∧ ∀ m : ℕ, m < 626 → Real.sqrt (m : ℝ) - Real.sqrt ((m : ℝ) - 1) ≥ 0.02) ↔ 
  (n = 626 ∧ Real.sqrt (n : ℝ) - Real.sqrt ((n : ℝ) - 1) < 0.02 ∧
   ∀ m : ℕ, m < n → Real.sqrt (m : ℝ) - Real.sqrt ((m : ℝ) - 1) ≥ 0.02) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l717_71734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_whale_tongue_weight_l717_71708

/-- The weight of an adult blue whale's tongue in pounds -/
def tongue_weight : ℚ := 6000

/-- The number of pounds in one ton -/
def pounds_per_ton : ℚ := 2000

/-- The weight of an adult blue whale's tongue in tons -/
def tongue_weight_in_tons : ℚ := tongue_weight / pounds_per_ton

theorem blue_whale_tongue_weight :
  tongue_weight_in_tons = 3 := by
  -- Unfold the definitions
  unfold tongue_weight_in_tons tongue_weight pounds_per_ton
  -- Perform the division
  norm_num

#eval tongue_weight_in_tons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_whale_tongue_weight_l717_71708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l717_71762

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 2*x else -x + 1/2

-- Define a, b, and c
noncomputable def a : ℝ := f ((1/2)^(1/3))
noncomputable def b : ℝ := Real.log (1/3) / Real.log (1/2)
noncomputable def c : ℝ := f ((1/3)^(1/2))

-- State the theorem
theorem a_b_c_relationship : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_relationship_l717_71762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l717_71784

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The problem statement -/
theorem y_coordinate_of_P (A B C D P : Point)
  (hA : A = ⟨-4, 0⟩)
  (hB : B = ⟨-3, 2⟩)
  (hC : C = ⟨3, 2⟩)
  (hD : D = ⟨4, 0⟩)
  (hP : distance P A + distance P D = distance P B + distance P C)
  (hDist : distance P A + distance P D = 10) :
  P.y = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l717_71784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_age_ratio_l717_71742

/-- Sarah's current age -/
noncomputable def S : ℝ := sorry

/-- Number of years ago when Sarah's age was five times the sum of her children's ages -/
noncomputable def M : ℝ := sorry

/-- Sum of the ages of Sarah's two children -/
noncomputable def children_sum : ℝ := S / 3

/-- Sarah's age M years ago -/
noncomputable def sarah_age_M_years_ago : ℝ := S - M

/-- Sum of the ages of Sarah's children M years ago -/
noncomputable def children_sum_M_years_ago : ℝ := children_sum - 2 * M

/-- The condition that Sarah's age M years ago was five times the sum of her children's ages then -/
axiom sarah_age_condition : sarah_age_M_years_ago = 5 * children_sum_M_years_ago

theorem sarah_age_ratio : S / M = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_age_ratio_l717_71742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_zero_l717_71768

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the angle opposite side c
noncomputable def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- The main theorem
theorem angle_C_is_zero (t : Triangle) 
  (h : (t.a + t.b + t.c) * (t.a + t.b - t.c) = 4 * t.a * t.b) : 
  angle_C t = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_zero_l717_71768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l717_71738

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- Function to check if a point is on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  ((p.x - e.center.x)^2 / e.semiMajorAxis^2) + ((p.y - e.center.y)^2 / e.semiMinorAxis^2) = 1

/-- Theorem: Minimum distance between points on given circle and ellipse -/
theorem min_distance_circle_ellipse :
  let circle : Circle := { center := { x := 0, y := 0 }, radius := 2 }
  let ellipse : Ellipse := { center := { x := 2, y := 0 }, semiMajorAxis := 3, semiMinorAxis := 5 }
  ∃ (p1 p2 : Point),
    isOnCircle p1 circle ∧
    isOnEllipse p2 ellipse ∧
    ∀ (q1 q2 : Point),
      isOnCircle q1 circle →
      isOnEllipse q2 ellipse →
      distance p1 p2 ≤ distance q1 q2 ∧
      distance p1 p2 = (Real.sqrt 163 - 6) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l717_71738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_in_two_circles_l717_71710

/-- Given two circles with specific properties, prove the possible lengths of a chord. -/
theorem chord_length_in_two_circles (r : ℝ) (O₁ O₂ : ℝ × ℝ) (ω₁ ω₂ : Set (ℝ × ℝ)) :
  (‖O₁ - O₂‖ = 5 * r) →  -- Distance between centers
  (ω₁ = {p : ℝ × ℝ | ‖p - O₁‖ = r}) →  -- Definition of ω₁
  (ω₂ = {p : ℝ × ℝ | ‖p - O₂‖ = 7 * r}) →  -- Definition of ω₂
  ∃ (A B M : ℝ × ℝ),
    (A ∈ ω₂ ∧ B ∈ ω₂) →  -- A and B are on ω₂
    (M ∈ ω₁) →  -- M is on ω₁
    (∃ t : ℝ, M = t • A + (1 - t) • B) →  -- M is on line AB
    (‖M - A‖ : ℝ) / (‖B - M‖ : ℝ) = 1 / 6 →  -- AM:MB = 1:6
    (‖A - B‖ = 7 * r * Real.sqrt 3 ∨ ‖A - B‖ = (7 * r * Real.sqrt 143) / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_in_two_circles_l717_71710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_inequality_1_f_inequality_2_l717_71799

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x - b * Real.exp x) / x

theorem f_properties (a b : ℝ) (ha : a ≠ 0) :
  (∀ x > 0, deriv (f a b) x = 0 → x = Real.exp 1) ∧
  (∃ x > 0, IsMinOn (f a b) (Set.Ioi 0) x) →
  a < 0 :=
sorry

theorem f_inequality_1 :
  ∀ x > 0, x * (f 1 1 x) + 2 < 0 :=
sorry

theorem f_inequality_2 :
  (∀ x > 1, x * (f 1 (-1) x) > Real.exp 1 + (1 + Real.exp 1) * (x - 1)) ∧
  ¬(∃ m > 1 + Real.exp 1, ∀ x > 1, x * (f 1 (-1) x) > Real.exp 1 + m * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_inequality_1_f_inequality_2_l717_71799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainder_pigeonhole_l717_71769

theorem square_remainder_pigeonhole (S : Finset ℤ) (h : S.card = 51) :
  ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x^2 % 100 = y^2 % 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_remainder_pigeonhole_l717_71769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_circle_separate_l717_71782

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the line y = kx + 1
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the circle x^2 + y^2 = 1/2
def circle_half (x y : ℝ) : Prop := x^2 + y^2 = 1/2

-- Define the point B
def point_B : ℝ × ℝ := (0, -2)

-- Define the isosceles triangle condition
def isosceles_triangle (B E F : ℝ × ℝ) : Prop :=
  let (xE, yE) := E
  let (xF, yF) := F
  (xE - B.1)^2 + (yE - B.2)^2 = (xF - B.1)^2 + (yF - B.2)^2

-- Main theorem
theorem ellipse_line_circle_separate :
  ∀ (k : ℝ) (E F : ℝ × ℝ),
  k ≠ 0 →
  ellipse_C (2*Real.sqrt 3) 1 →
  (∃ (x y : ℝ), ellipse_C x y ∧ y < 0 ∧ x = 0) →
  (∃ (xE yE : ℝ), E = (xE, yE) ∧ ellipse_C xE yE ∧ line k xE yE) →
  (∃ (xF yF : ℝ), F = (xF, yF) ∧ ellipse_C xF yF ∧ line k xF yF) →
  E ≠ F →
  isosceles_triangle point_B E F →
  (∀ (x y : ℝ), (y = (-1/(4*k))*x + 1) → ¬(circle_half x y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_circle_separate_l717_71782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l717_71722

/-- Given a circle with x-intercepts a and b, and y-intercept c (c ≠ 0),
    its equation is x² + y² - (a+b)x - (c + ab/c)y + ab = 0 -/
theorem circle_equation (a b c : ℝ) (hc : c ≠ 0) :
  let f (x y : ℝ) := x^2 + y^2 - (a+b)*x - (c + a*b/c)*y + a*b
  ∀ x y, ((x = a ∨ x = b) ∧ y = 0 → f x y = 0) ∧
         (x = 0 ∧ y = c → f x y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l717_71722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l717_71721

noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x) / Real.log 10

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioi 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l717_71721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l717_71798

noncomputable def z : ℂ := Complex.I / (1 + Complex.I) + (1 + Real.sqrt 3 * Complex.I) ^ 2

theorem z_in_second_quadrant : Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l717_71798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2_to_100_l717_71735

-- Define the sequence b
def b : ℕ → ℕ
| 0 => 2  -- Base case for n = 0 (which corresponds to b₁ in the original problem)
| n + 1 => if n % 2 = 0 then ((n + 1) / 2) * b ((n + 1) / 2)  -- condition (ii)
           else b n + 1  -- condition (iii)

-- Lemma: b(2^n) = 2^(n+1)
lemma b_power_of_two (n : ℕ) : b (2^n) = 2^(n+1) := by
  sorry

-- Main theorem
theorem b_2_to_100 : b (2^100) = 2^101 := by
  sorry

#eval b (2^100)  -- This will evaluate b(2^100) if possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2_to_100_l717_71735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l717_71750

/-- A power function that passes through the point (√2, 2√2) -/
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

/-- The point through which the function passes -/
noncomputable def point : ℝ × ℝ := (Real.sqrt 2, 2 * Real.sqrt 2)

theorem power_function_through_point :
  ∃ α : ℝ, f α (point.1) = point.2 ∧ α = 3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l717_71750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_land_division_steps_l717_71785

/-- Represents the number of times a parcel is divided into 3 parts -/
def x : ℕ := sorry

/-- Represents the number of times a parcel is divided into 4 parts -/
def y : ℕ := sorry

/-- Represents the total number of steps performed -/
def n : ℕ := sorry

/-- The number of obtained parcels equals four times the number of steps made -/
axiom parcel_equation : 5 + 2 * x + 3 * y = 4 * n

/-- The total number of steps is the sum of divisions into 3 and 4 parts -/
axiom step_equation : n = x + y

/-- The solution satisfies the equation 5 = 2x + y -/
axiom solution_equation : 5 = 2 * x + y

theorem land_division_steps : n = 3 ∨ n = 4 ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_land_division_steps_l717_71785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l717_71715

/-- Calculates the initial amount of water in a bowl given evaporation conditions -/
theorem initial_water_amount 
  (evaporation_rate : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) : 
  evaporation_rate = 0.004 ∧ 
  evaporation_period = 50 ∧ 
  evaporation_percentage = 0.02 →
  (evaporation_rate * ↑evaporation_period) / evaporation_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l717_71715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_special_integers_l717_71747

/-- Represents a sequence of integers with special properties -/
def SpecialSequence : ℕ → ℕ := sorry

/-- Checks if a natural number has no zeros in its decimal representation -/
def hasNoZeros (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

/-- Calculates the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating the existence of infinitely many integers with special properties -/
theorem infinite_special_integers :
  ∃ (a : ℕ → ℕ), 
    (∀ n : ℕ, n > 0 → hasNoZeros (a n)) ∧ 
    (∀ n : ℕ, n > 0 → (a n) % (digitSum (a n)) = 0) ∧
    (∀ m : ℕ, ∃ n : ℕ, n > m ∧ a n ≠ 0) :=
by
  sorry

#check infinite_special_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_special_integers_l717_71747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_divisible_by_three_and_digit_product_l717_71778

def numbers : List Nat := [3543, 3552, 3567, 3579, 3581]

def is_divisible_by_three (n : Nat) : Bool :=
  n % 3 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem unique_non_divisible_by_three_and_digit_product :
  ∃! n, n ∈ numbers ∧ ¬is_divisible_by_three n ∧ 
    units_digit n * tens_digit n = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_divisible_by_three_and_digit_product_l717_71778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_M_and_R_l717_71749

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (E F R G H : ℝ × ℝ)

-- Define the arcs and their measures
noncomputable def arc_FR : ℝ := 60
noncomputable def arc_RG : ℝ := 50

-- Define the angles M and R
noncomputable def angle_M : ℝ := (arc_FR + arc_RG - (360 - arc_FR - arc_RG)) / 2
noncomputable def angle_R : ℝ := (360 - arc_FR - arc_RG) / 2

-- Theorem statement
theorem sum_of_angles_M_and_R :
  E ∈ circle → F ∈ circle → R ∈ circle → G ∈ circle → H ∈ circle →
  angle_M + angle_R = 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_M_and_R_l717_71749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_age_l717_71743

-- Define Alice's age and Bob's age as variables
variable (alice_age : ℝ)
variable (bob_age : ℝ)

-- Define the conditions as axioms
axiom condition1 : bob_age = 3 * alice_age - 20
axiom condition2 : bob_age + alice_age = 70

-- State the theorem to prove
theorem bobs_age : bob_age = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_age_l717_71743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l717_71752

-- Define the line l: x + y = 2
def line (x y : ℝ) : Prop := x + y = 2

-- Define the circle C: x^2 - 2y = 3
def circle_C (x y : ℝ) : Prop := x^2 - 2*y = 3

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l717_71752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_time_calculation_l717_71751

-- Define the constants from the problem
def speed : ℝ := 10  -- miles per hour
def initial_time : ℝ := 30  -- minutes
def second_distance : ℝ := 15  -- miles
def final_distance : ℝ := 20  -- miles
def total_time : ℝ := 270  -- minutes

-- Define the rest time as a function
def rest_time : ℝ → Prop := λ t => 
  t = total_time - 
    (initial_time + 
     (second_distance / speed * 60) + 
     (final_distance / speed * 60)) ∧
  t = 30

-- Theorem statement
theorem rest_time_calculation : 
  ∃ t, rest_time t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rest_time_calculation_l717_71751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_composite_function_l717_71714

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else Real.exp (x * Real.log 2)

-- State the theorem
theorem range_of_composite_function :
  {a : ℝ | f (f a) = Real.exp ((f a) * Real.log 2)} = {a : ℝ | a ≥ 2/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_composite_function_l717_71714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_V₃_l717_71728

def horner_polynomial (x : ℤ) : List ℤ → ℤ
| [] => 0
| a :: coeffs => a + x * horner_polynomial x coeffs

def V₃ (x : ℤ) : ℤ := horner_polynomial x [23, -8, 10, -3]

theorem horner_method_V₃ :
  V₃ (-4) = -49 :=
by
  -- Expand the definition of V₃
  unfold V₃
  -- Expand the definition of horner_polynomial
  unfold horner_polynomial
  -- Perform the computation
  simp [Int.mul_add, Int.add_mul, Int.mul_assoc]
  -- The rest of the proof
  sorry

#eval V₃ (-4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_V₃_l717_71728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l717_71718

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then k / (x + 1)
  else if x < 0 then -k / (-x + 1)
  else 0

theorem f_properties (k : ℝ) (h : k ≠ 0) :
  -- f is odd
  (∀ x, f k (-x) = -(f k x)) ∧
  -- Part 1: When k = 1, f has the specified form
  (k = 1 →
    (∀ x, x < 0 → f k x = 1 / (x - 1)) ∧
    (f k 0 = 0) ∧
    (∀ x, x > 0 → f k x = 1 / (x + 1))) ∧
  -- Part 2: If f(x) > 1 for 0 < x < 1, then k ≥ 2
  ((∀ x, 0 < x → x < 1 → f k x > 1) → k ≥ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l717_71718
