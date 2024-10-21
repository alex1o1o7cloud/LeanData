import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l201_20186

def M : Set ℚ := {-1, 0, 1, 2, 3}
def N : Set ℚ := {x | x^2 - 2*x ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l201_20186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_discount_percentage_l201_20178

/-- Calculates the overall discount percentage for four items with given markups and profits -/
theorem overall_discount_percentage
  (markup1 markup2 markup3 markup4 : ℝ)
  (profit1 profit2 profit3 profit4 : ℝ)
  (h1 : markup1 = 0.50)
  (h2 : markup2 = 0.75)
  (h3 : markup3 = 1.00)
  (h4 : markup4 = 1.50)
  (h5 : profit1 = 0.125)
  (h6 : profit2 = 0.225)
  (h7 : profit3 = 0.30)
  (h8 : profit4 = 0.50)
  (purchase_cost : ℝ)
  (h9 : purchase_cost > 0) :
  let marked_price1 := purchase_cost * (1 + markup1)
  let marked_price2 := purchase_cost * (1 + markup2)
  let marked_price3 := purchase_cost * (1 + markup3)
  let marked_price4 := purchase_cost * (1 + markup4)
  let selling_price1 := purchase_cost * (1 + profit1)
  let selling_price2 := purchase_cost * (1 + profit2)
  let selling_price3 := purchase_cost * (1 + profit3)
  let selling_price4 := purchase_cost * (1 + profit4)
  let total_marked_price := marked_price1 + marked_price2 + marked_price3 + marked_price4
  let total_discount := (marked_price1 - selling_price1) + (marked_price2 - selling_price2) +
                        (marked_price3 - selling_price3) + (marked_price4 - selling_price4)
  let overall_discount_percentage := (total_discount / total_marked_price) * 100
  abs (overall_discount_percentage - 33.55) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_discount_percentage_l201_20178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l201_20145

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a ≠ 0 ∨ b ≠ 0)

/-- The point (1, -2) --/
def P : ℝ × ℝ := (1, -2)

/-- A line passes through a point --/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- The x-intercept of a line --/
noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.a

/-- The y-intercept of a line --/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

/-- The intercepts are opposite numbers --/
def opposite_intercepts (l : Line) : Prop :=
  x_intercept l = -(y_intercept l)

theorem line_equation (l : Line) :
  passes_through l P ∧ opposite_intercepts l →
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -3) ∨ (l.a = 2 ∧ l.b = 1 ∧ l.c = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l201_20145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_relation_l201_20175

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define perpendicular relation between two planes
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_relation 
  (l : Line) (m : Line) (α β γ : Plane)
  (distinct_lines : l ≠ m)
  (distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h1 : perpendicular l γ)
  (h2 : plane_perpendicular α γ) :
  parallel l α ∨ contained_in l α :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_relation_l201_20175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l201_20171

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.rpow 5 (-x) else Real.rpow 5 x - 1

-- State the theorem
theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l201_20171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_a_value_l201_20117

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x) / (1 + a * (2^x))

theorem symmetric_function_a_value (a : ℝ) :
  (∀ x, f a x + f a (-x) = 1) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_a_value_l201_20117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l201_20134

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (7 + 6*x - x^2)

-- State the theorem
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 ≤ x ∧ x ≤ 7} :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l201_20134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_formula_problems_l201_20154

theorem multiplication_formula_problems :
  (2023^2 - 2024 * 2022 = 1) ∧
  (∀ y : ℝ, (y/2 - 7)^2 - (y/2 + 7)^2 = -14*y) := by
  constructor
  · -- Proof for the first part
    sorry
  · -- Proof for the second part
    intro y
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_formula_problems_l201_20154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_Q_l201_20106

-- Define the points Q and Q'
def Q : ℝ × ℝ := (-3, 1)
def Q' : ℝ × ℝ := (3, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem distance_Q_to_Q'_is_6 : distance Q Q' = 6 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp [Q, Q']
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_Q_l201_20106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l201_20169

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi/3) - 1

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2*x - 2*Real.pi/3) - 1

theorem axis_of_symmetry (x : ℝ) :
  g (Real.pi/12 + x) = g (Real.pi/12 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l201_20169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_correct_l201_20196

noncomputable def line1 (x : ℝ) : ℝ := 2 * x
noncomputable def line2 (x : ℝ) : ℝ := -x + 2

noncomputable def intersection_x : ℝ := 2/3
noncomputable def intersection_y : ℝ := 4/3

noncomputable def slope_bisector : ℝ := (1 + Real.sqrt 6) / 3
noncomputable def intercept_bisector : ℝ := (6 - 2 * Real.sqrt 6) / 9

noncomputable def angle_bisector (x : ℝ) : ℝ := slope_bisector * x + intercept_bisector

theorem angle_bisector_correct :
  (∀ x, line1 x = 2 * x) ∧
  (∀ x, line2 x = -x + 2) ∧
  (line1 intersection_x = intersection_y) ∧
  (line2 intersection_x = intersection_y) ∧
  (∀ x, angle_bisector x = slope_bisector * x + intercept_bisector) →
  angle_bisector intersection_x = intersection_y ∧
  (slope_bisector > (line1 1 - line1 0)) ∧
  (slope_bisector < (line2 1 - line2 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_correct_l201_20196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_rotation_area_l201_20120

/- Given a semicircle with radius R -/
variable (R : ℝ)

/- Define the rotation angle in radians -/
noncomputable def α : ℝ := 30 * (Real.pi / 180)

/- Define the area of the shaded figure -/
noncomputable def shaded_area (R : ℝ) : ℝ := (Real.pi * R^2) / 3

/- Theorem statement -/
theorem semicircle_rotation_area (R : ℝ) :
  shaded_area R = (Real.pi * R^2) / 3 := by
  -- Unfold the definition of shaded_area
  unfold shaded_area
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_rotation_area_l201_20120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_problem_solution_l201_20136

/-- The time taken for two people cycling towards each other to meet -/
noncomputable def meeting_time (distance : ℝ) (speed_sum : ℝ) : ℝ :=
  distance / speed_sum

/-- Theorem: The meeting time is correct given the initial distance and sum of speeds -/
theorem meeting_time_correct (distance : ℝ) (speed_sum : ℝ) 
  (h1 : distance > 0) (h2 : speed_sum > 0) :
  meeting_time distance speed_sum = distance / speed_sum :=
by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- The result follows directly from the definition
  rfl

/-- Corollary: For the given problem, the meeting time is 2 hours -/
theorem problem_solution :
  meeting_time 65 32.5 = 2 :=
by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_correct_problem_solution_l201_20136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l201_20113

/-- Given a 15% reduction in oil price, proves that if 4 kg more can be bought for Rs. 1200 after reduction, then the reduced price is approximately Rs. 45 per kg. -/
theorem oil_price_reduction (original_price : ℝ) (original_quantity : ℝ) : 
  original_price > 0 →
  original_quantity > 0 →
  original_price * original_quantity = 1200 →
  (original_quantity + 4) * (original_price * 0.85) = 1200 →
  ∃ (reduced_price : ℝ), (reduced_price ≥ 44.99 ∧ reduced_price ≤ 45.01) ∧ reduced_price = original_price * 0.85 :=
by
  intro h_pos_price h_pos_quantity h_original h_reduced
  -- Define the reduced price
  let reduced_price := original_price * 0.85
  
  -- Prove that reduced_price exists and satisfies the conditions
  use reduced_price
  apply And.intro
  · -- Prove that reduced_price is approximately 45
    sorry -- This part requires more detailed calculations
  · -- Prove that reduced_price = original_price * 0.85
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l201_20113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blind_box_problem_l201_20162

/-- A blind box set with a given number of items and probability of getting a small rabbit toy -/
structure BlindBoxSet where
  numItems : ℕ
  rabbitProb : ℚ

/-- The number of small rabbit toys obtained from buying multiple sets of a given blind box set -/
def numRabbits (n : ℕ) (set : BlindBoxSet) : Real :=
  n * set.rabbitProb

theorem blind_box_problem (setA setB : BlindBoxSet)
    (hA : setA = ⟨4, 1/4⟩) (hB : setB = ⟨2, 1/2⟩) :
  let ξ := numRabbits 3 setB
  let mixedProb := (4 * (1/4) + 2 * (1/2)) / 6
  -- 1. Distribution of ξ (approximated as probabilities)
  (ξ = 3/2) ∧
  -- 2. Mathematical expectation of ξ
  (ξ = 3/2) ∧
  -- 3. Probability that a randomly selected small rabbit toy came from set B
  ((1/2 * 2/6) / mixedProb = 1/3) := by
  sorry

#check blind_box_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blind_box_problem_l201_20162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l201_20127

noncomputable def asymptote_slope (a b : ℝ) : ℝ := b / a

def is_hyperbola (x y a b h k : ℝ) : Prop :=
  ((y - k)^2 / b^2) - ((x - h)^2 / a^2) = 1

theorem hyperbola_asymptote_slope :
  ∀ (x y : ℝ),
  is_hyperbola x y 3 4 (-2) 1 →
  asymptote_slope 3 4 = 4/3 ∨ asymptote_slope 3 4 = -4/3 := by
  sorry

#check hyperbola_asymptote_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l201_20127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_field_dimensions_l201_20114

/-- Represents the dimensions of a rectangular field -/
structure RectField where
  length : ℝ
  width : ℝ

/-- The total fence length available -/
def totalFence : ℝ := 1200

/-- Calculates the perimeter of the field (excluding the river side) -/
def fenceUsed (f : RectField) : ℝ := 2 * f.length + f.width

/-- Calculates the area of the field -/
def area (f : RectField) : ℝ := f.length * f.width

/-- Theorem stating that the sum of length and width for the maximum area field is 900 -/
theorem max_area_field_dimensions :
  ∃ (f : RectField), fenceUsed f = totalFence ∧ 
    (∀ (g : RectField), fenceUsed g = totalFence → area g ≤ area f) ∧
    f.length + f.width = 900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_field_dimensions_l201_20114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_cost_is_85_l201_20194

/-- The cost of a classic gerbera in korunas -/
def classic_gerbera_cost : ℚ := sorry

/-- The cost of a mini gerbera in korunas -/
def mini_gerbera_cost : ℚ := sorry

/-- The cost of a decorative ribbon in korunas -/
def ribbon_cost : ℚ := sorry

/-- The total cost of 5 classic gerberas and 1 ribbon is 295 korunas -/
axiom classic_bouquet_cost : 5 * classic_gerbera_cost + ribbon_cost = 295

/-- The total cost of 7 mini gerberas and 1 ribbon is 295 korunas -/
axiom mini_bouquet_cost : 7 * mini_gerbera_cost + ribbon_cost = 295

/-- The total cost of 2 mini gerberas and 1 classic gerbera is 102 korunas -/
axiom mixed_bouquet_cost : 2 * mini_gerbera_cost + classic_gerbera_cost = 102

/-- The cost of one ribbon is 85 korunas -/
theorem ribbon_cost_is_85 : ribbon_cost = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_cost_is_85_l201_20194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_dessert_menus_l201_20105

/-- Represents the types of desserts available. -/
inductive Dessert
  | Cake
  | Pie
  | IceCream
  | Pudding

/-- Represents the days of the week. -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- A function that returns the number of dessert choices for a given day. -/
def dessertChoices (day : Day) : ℕ :=
  match day with
  | Day.Sunday => 4
  | Day.Monday => 1
  | Day.Tuesday => 3
  | Day.Wednesday => 3
  | Day.Thursday => 3
  | Day.Friday => 1
  | Day.Saturday => 3

/-- Theorem stating that the number of different dessert menus for the week is 324. -/
theorem number_of_dessert_menus :
  (List.prod (List.map dessertChoices [Day.Sunday, Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday])) = 324 := by
  sorry

/-- Lemma stating that ice cream must be served on Monday. -/
lemma ice_cream_on_monday (menu : Day → Dessert) :
  menu Day.Monday = Dessert.IceCream := by
  sorry

/-- Lemma stating that cake must be served on Friday. -/
lemma cake_on_friday (menu : Day → Dessert) :
  menu Day.Friday = Dessert.Cake := by
  sorry

/-- Lemma stating that the same dessert cannot be served on consecutive days. -/
lemma no_consecutive_desserts (menu : Day → Dessert) :
  ∀ d₁ d₂ : Day, (d₁ ≠ d₂) → (menu d₁ = menu d₂ → False) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_dessert_menus_l201_20105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_frosting_needs_l201_20165

/-- Represents the amount of frosting needed for different baked goods -/
structure FrostingNeeds where
  layer_cake : ℚ
  single_cake : ℚ
  brownies : ℚ
  cupcakes : ℚ

/-- Represents the quantities of different baked goods -/
structure BakedGoods where
  layer_cakes : ℕ
  single_cakes : ℕ
  brownies : ℕ
  cupcake_dozens : ℕ

/-- Calculates the total cans of frosting needed -/
def total_frosting_cans (needs : FrostingNeeds) (goods : BakedGoods) : ℚ :=
  needs.layer_cake * goods.layer_cakes +
  needs.single_cake * goods.single_cakes +
  needs.brownies * goods.brownies +
  needs.cupcakes * goods.cupcake_dozens

/-- Theorem: Paul needs 21 cans of frosting -/
theorem paul_frosting_needs : 
  let needs : FrostingNeeds := {
    layer_cake := 1,
    single_cake := 1/2,
    brownies := 1/2,
    cupcakes := 1/2
  }
  let goods : BakedGoods := {
    layer_cakes := 3,
    single_cakes := 12,
    brownies := 18,
    cupcake_dozens := 6
  }
  total_frosting_cans needs goods = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_frosting_needs_l201_20165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l201_20167

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^3 - 8 else (- x)^3 - 8

-- State the theorem
theorem solution_set_f (x : ℝ) : 
  (∀ y : ℝ, f y = f (-y)) →  -- f is even
  (∀ y : ℝ, y ≥ 0 → f y = y^3 - 8) →  -- f(x) = x^3 - 8 for x ≥ 0
  (f (x - 2) > 0 ↔ x < 0 ∨ x > 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l201_20167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_approx_l201_20122

-- Define the expression as noncomputable
noncomputable def f : ℝ := Real.sqrt ((9^6 + 3^12) / (9^3 + 3^17))

-- State the theorem
theorem f_approx : ∃ ε > 0, |f - 0.091| < ε := by
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_approx_l201_20122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l201_20183

/-- The function to be maximized -/
noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

/-- The domain constraints -/
def domain (x y : ℝ) : Prop :=
  1/4 ≤ x ∧ x ≤ 3/5 ∧ 1/5 ≤ y ∧ y ≤ 1/2

theorem max_value_of_f :
  ∃ (x y : ℝ), domain x y ∧ f x y = 2/5 ∧ ∀ (x' y' : ℝ), domain x' y' → f x' y' ≤ 2/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l201_20183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_dance_ratio_l201_20188

theorem school_dance_ratio (total : ℕ) (faculty_percent : ℚ) (boys : ℕ) : 
  total = 100 →
  faculty_percent = 1/10 →
  boys = 30 →
  (total - (faculty_percent * ↑total).floor - boys : ℚ) / boys = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_dance_ratio_l201_20188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_edge_length_is_three_halves_l201_20180

/-- The edge length of a cube inscribed in a cone -/
noncomputable def inscribedCubeEdgeLength (coneHeight : ℝ) (coneBaseRadius : ℝ) : ℝ :=
  3 / 2

/-- Theorem: The edge length of a cube inscribed in a cone with height 6 and base radius √2 is 3/2 -/
theorem inscribed_cube_edge_length_is_three_halves :
  inscribedCubeEdgeLength 6 (Real.sqrt 2) = 3 / 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_edge_length_is_three_halves_l201_20180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_equals_two_l201_20156

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log (x + 2) / Real.log 3 + a else Real.exp x - 1

theorem f_of_a_equals_two (a : ℝ) :
  (f a (f a (Real.log 2)) = 2 * a) → f a a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_equals_two_l201_20156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antisymmetric_real_squared_zero_is_zero_odd_antisymmetric_adjoint_squared_zero_l201_20115

variable {n : ℕ}

def antisymmetric {R : Type*} [Ring R] (A : Matrix (Fin n) (Fin n) R) : Prop :=
  A = -A.transpose

theorem antisymmetric_real_squared_zero_is_zero
  (A : Matrix (Fin n) (Fin n) ℝ)
  (h_antisym : antisymmetric A)
  (h_squared_zero : A ^ 2 = 0) :
  A = 0 := by sorry

theorem odd_antisymmetric_adjoint_squared_zero
  (h_odd : Odd n)
  (A : Matrix (Fin n) (Fin n) ℂ)
  (h_antisym : antisymmetric A)
  (B : Matrix (Fin n) (Fin n) ℂ)
  (h_adjoint : A = Matrix.adjugate B) :
  A ^ 2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antisymmetric_real_squared_zero_is_zero_odd_antisymmetric_adjoint_squared_zero_l201_20115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l201_20142

noncomputable def f (x : ℝ) : ℝ := (1/4)^(x-1) - 4*(1/2)^x + 2

theorem f_min_max :
  (∀ x ∈ Set.Icc (0 : ℝ) 2, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = 1) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) 2, f x ≤ 2) ∧
  (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l201_20142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_arrangement_l201_20133

-- Define the daily picking capacity of a novice tea picker
def novice_capacity : ℚ := 10

-- Define the daily picking capacity of a skilled tea picker
def skilled_capacity : ℚ := 3 * novice_capacity

-- Define the daily wage of a skilled tea picker
def skilled_wage : ℚ := 300

-- Define the daily wage of a novice tea picker
def novice_wage : ℚ := 80

-- Define the total amount of tea to be picked in one day
def total_tea : ℚ := 600

-- Define the maximum number of skilled tea pickers available
def max_skilled : ℕ := 20

-- Define the maximum number of novice tea pickers available
def max_novice : ℕ := 15

-- Define the cost function
def cost (skilled : ℕ) (novice : ℕ) : ℚ :=
  skilled * skilled_wage + novice * novice_wage

-- Define the tea picked function
def tea_picked (skilled : ℕ) (novice : ℕ) : ℚ :=
  (skilled : ℚ) * skilled_capacity + (novice : ℚ) * novice_capacity

-- Theorem stating the optimal arrangement
theorem optimal_arrangement :
  ∃ (skilled novice : ℕ),
    skilled ≤ max_skilled ∧
    novice ≤ max_novice ∧
    tea_picked skilled novice ≥ total_tea ∧
    ∀ (s n : ℕ),
      s ≤ max_skilled →
      n ≤ max_novice →
      tea_picked s n ≥ total_tea →
      cost skilled novice ≤ cost s n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_arrangement_l201_20133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_min_value_is_three_min_value_achieved_l201_20193

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(x-3) = (1/2 : ℝ)^y) : 
  ∀ a b : ℝ, a > 0 → b > 0 → (2 : ℝ)^(a-3) = (1/2 : ℝ)^b → 1/x + 4/y ≤ 1/a + 4/b :=
by sorry

theorem min_value_is_three (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(x-3) = (1/2 : ℝ)^y) : 
  1/x + 4/y ≥ 3 :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(x-3) = (1/2 : ℝ)^y) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (2 : ℝ)^(a-3) = (1/2 : ℝ)^b ∧ 1/a + 4/b = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_min_value_is_three_min_value_achieved_l201_20193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mini_train_speed_approx_l201_20164

/-- The speed of a mini-train in kilometers per hour -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that the speed of the mini-train is approximately 75 kmph -/
theorem mini_train_speed_approx :
  let length : ℝ := 62.505
  let time : ℝ := 3
  abs (train_speed length time - 75) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mini_train_speed_approx_l201_20164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_g_l201_20138

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := 1 - f (x + 2)

-- State the theorem
theorem domain_and_range_of_g :
  (∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 0 1) →
  (Set.Icc 0 3).image f = Set.Icc 0 1 →
  (Set.Icc (-2) 1).image g = Set.Icc 0 1 ∧
  ∀ x, g x ∈ Set.Icc 0 1 → x ∈ Set.Icc (-2) 1 := by
  sorry

#check domain_and_range_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_range_of_g_l201_20138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_expression_eval_l201_20135

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem prime_expression_eval :
  ∀ x y : ℤ,
    is_prime x.toNat →
    is_prime y.toNat →
    x ≠ y →
    20 < x →
    x < 30 →
    20 < y →
    y < 30 →
    x * y - (x + y) - (x^2 + y^2) = -755 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_expression_eval_l201_20135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zeros_inequality_l201_20158

/-- Given positive real numbers a and b, if f(x) = x^3 + ax^2 + 2bx - 1 has three distinct zeros
    and g(x) = 2x^2 + 2bx + a has no zeros, then a - b > 1 -/
theorem polynomial_zeros_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hf : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (fun t ↦ t^3 + a*t^2 + 2*b*t - 1) x = 0 ∧
    (fun t ↦ t^3 + a*t^2 + 2*b*t - 1) y = 0 ∧
    (fun t ↦ t^3 + a*t^2 + 2*b*t - 1) z = 0)
  (hg : ∀ x : ℝ, 2*x^2 + 2*b*x + a ≠ 0) : 
  a - b > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zeros_inequality_l201_20158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_is_100_l201_20126

-- Define the rate of the old machine (in bolts per hour)
noncomputable def old_machine_rate : ℝ := sorry

-- Define the rate of the new machine (in bolts per hour)
noncomputable def new_machine_rate : ℝ := 150

-- Define the time both machines work together (in hours)
noncomputable def work_time : ℝ := 96 / 60

-- Define the total number of bolts produced
noncomputable def total_bolts : ℝ := 400

-- Theorem statement
theorem old_machine_rate_is_100 :
  (old_machine_rate + new_machine_rate) * work_time = total_bolts →
  old_machine_rate = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_old_machine_rate_is_100_l201_20126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_l201_20190

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem symmetry_axis_of_sine (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) →
  (∃ k : ℤ, f ω (Real.pi / 6) = f ω (k * Real.pi - Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_l201_20190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l201_20199

-- Define the function f
noncomputable def f : ℝ → ℝ :=
  fun x => if x > 0 then x^3 + x + 1 else -x^3 - x + 1

-- State the theorem
theorem f_is_even_and_correct : 
  (∀ x, f x = f (-x)) ∧ 
  (∀ x > 0, f x = x^3 + x + 1) ∧ 
  (∀ x < 0, f x = -x^3 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l201_20199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_repeating_decimal_expression_l201_20101

/-- Represents a repeating decimal with non-repeating parts A and B, and repeating part C -/
structure RepeatingDecimal where
  A : ℕ  -- First non-repeating part
  B : ℕ  -- Second non-repeating part
  C : ℕ  -- Repeating part
  u : ℕ  -- Number of digits in A
  v : ℕ  -- Number of digits in B
  w : ℕ  -- Number of digits in C

/-- The value of the repeating decimal as a real number -/
noncomputable def RepeatingDecimal.value (X : RepeatingDecimal) : ℝ :=
  (X.A * (10 : ℝ)^(X.v + X.w) + X.B * (10 : ℝ)^X.w + X.C) / ((10 : ℝ)^(X.u + X.v + X.w) - (10 : ℝ)^(X.u + X.v))

/-- Theorem stating that the given expression is incorrect for repeating decimals -/
theorem incorrect_repeating_decimal_expression (X : RepeatingDecimal) :
  (10 : ℝ)^(X.u + X.v) * ((10 : ℝ)^X.w - 1) * X.value ≠ X.B * (10 : ℝ)^X.v + X.C * (X.A - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_repeating_decimal_expression_l201_20101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coincides_implies_equilateral_l201_20153

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  (((t.A.1 + t.B.1 + t.C.1) / 3), ((t.A.2 + t.B.2 + t.C.2) / 3))

-- Define the centroid of the triangle's boundary
noncomputable def boundary_centroid (t : Triangle) : ℝ × ℝ :=
  (((t.A.1 + t.B.1 + t.C.1) / 3), ((t.A.2 + t.B.2 + t.C.2) / 3))

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  let d_AB := (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2
  let d_BC := (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2
  let d_CA := (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2
  d_AB = d_BC ∧ d_BC = d_CA

-- Theorem statement
theorem centroid_coincides_implies_equilateral (t : Triangle) :
  centroid t = boundary_centroid t → is_equilateral t :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_coincides_implies_equilateral_l201_20153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_scalar_multiplication_l201_20152

theorem vector_subtraction_scalar_multiplication (a b : Fin 3 → ℝ) :
  a = ![3, -2, 5] →
  b = ![-1, 4, 0] →
  a - 4 • b = ![7, -18, 5] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_scalar_multiplication_l201_20152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducibility_l201_20125

theorem fraction_irreducibility (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducibility_l201_20125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l201_20109

/-- The present value of a machine given its future value and depreciation rate. -/
noncomputable def present_value (future_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  future_value / ((1 - depreciation_rate) ^ years)

/-- Theorem stating the present value of a machine given specific conditions. -/
theorem machine_present_value :
  let future_value : ℝ := 36100
  let depreciation_rate : ℝ := 0.05
  let years : ℕ := 2
  let calculated_present_value := present_value future_value depreciation_rate years
  ∃ ε > 0, abs (calculated_present_value - 39978.95) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l201_20109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_nested_sqrt_l201_20185

-- Define the nested square root expression as noncomputable
noncomputable def nestedSqrt : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))

-- State the theorem
theorem fourth_power_nested_sqrt :
  nestedSqrt^4 = 10 + 4 * Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)) + 2 * Real.sqrt (2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_nested_sqrt_l201_20185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l201_20160

def purchase_price : ℕ := 42000
def engine_repairs : ℕ := 5000
def bodywork_repairs : ℕ := 3000
def paint_job : ℕ := 2500
def tires_and_wheels : ℕ := 4000
def interior_upgrades : ℕ := 1500
def selling_price : ℕ := 65000

def total_cost : ℕ := purchase_price + engine_repairs + bodywork_repairs + paint_job + tires_and_wheels + interior_upgrades

def profit : ℤ := selling_price - total_cost

noncomputable def profit_percentage : ℝ := (profit : ℝ) / (total_cost : ℝ) * 100

theorem profit_percentage_approx :
  abs (profit_percentage - 12.07) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l201_20160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l201_20112

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l201_20112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_24_l201_20170

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x * (2 - x) else -(-x * (2 - (-x)))

-- State the theorem
theorem f_of_4_equals_24 :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x, x < 0 → f x = x * (2 - x)) →  -- definition for x < 0
  f 4 = 24 := by
  intros h_odd h_def
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_24_l201_20170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_X_to_Y_l201_20172

/-- Represents a country with its population and wealth percentages -/
structure Country where
  population_percent : ℚ
  wealth_percent : ℚ

/-- The world consisting of three countries -/
structure World where
  X : Country
  Y : Country
  Z : Country

/-- Calculate the ratio of wealth per citizen between two countries -/
noncomputable def wealth_per_citizen_ratio (w : World) : ℚ :=
  (w.X.wealth_percent * w.Y.population_percent) / (w.X.population_percent * w.Y.wealth_percent)

/-- Theorem stating the ratio of wealth per citizen between countries X and Y -/
theorem wealth_ratio_X_to_Y (w : World) : 
  wealth_per_citizen_ratio w = (w.X.wealth_percent * w.Y.population_percent) / (w.X.population_percent * w.Y.wealth_percent) := by
  -- Unfold the definition of wealth_per_citizen_ratio
  unfold wealth_per_citizen_ratio
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_X_to_Y_l201_20172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_parallel_vectors_l201_20173

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem tan_alpha_parallel_vectors (α : ℝ) :
  let a : ℝ × ℝ := (Real.sin α, 2)
  let b : ℝ × ℝ := (-Real.cos α, 1)
  parallel a b → Real.tan α = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_parallel_vectors_l201_20173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_invalid_for_nonperpendicular_axes_l201_20111

/-- A geometric shape with symmetry axes -/
structure SymmetricShape where
  /-- The set of symmetry axes of the shape -/
  symmetryAxes : Set (ℝ × ℝ)
  /-- Axiom: The shape has at least two distinct symmetry axes -/
  twoAxes : ∃ (a b : ℝ × ℝ), a ∈ symmetryAxes ∧ b ∈ symmetryAxes ∧ a ≠ b

/-- Definition: Axes are perpendicular -/
def areAxesPerpendicular (s : SymmetricShape) : Prop :=
  ∀ (a b : ℝ × ℝ), a ∈ s.symmetryAxes → b ∈ s.symmetryAxes → a ≠ b →
    (a.1 * b.1 + a.2 * b.2) = 0

/-- The theorem from the previous question (hypothetical) -/
def previousTheorem (s : SymmetricShape) : Prop :=
  sorry -- The actual theorem statement would go here

/-- There exists a shape with non-perpendicular symmetry axes where the previous theorem doesn't hold -/
theorem theorem_invalid_for_nonperpendicular_axes :
  ∃ (s : SymmetricShape), ¬(areAxesPerpendicular s) ∧ ¬(previousTheorem s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_invalid_for_nonperpendicular_axes_l201_20111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l201_20155

theorem cosine_sum_identity : 
  (Real.cos (75 * π / 180))^2 + (Real.cos (15 * π / 180))^2 + 
  (Real.cos (75 * π / 180)) * (Real.cos (15 * π / 180)) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_identity_l201_20155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_squared_plus_b_squared_l201_20166

/-- Given that the coefficient of x³ in the expansion of (ax² + b/x)⁶ is 20,
    prove that the minimum value of a² + b² is 2 -/
theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∃ k, k * a^3 * b^3 = 20) → 
  (∀ x y : ℝ, x^2 + y^2 ≥ 2 * x * y) → 
  (a * b = 1) → 
  (∀ x y : ℝ, x^2 + y^2 ≥ a^2 + b^2) → 
  a^2 + b^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_squared_plus_b_squared_l201_20166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l201_20102

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculate the distance from a point to a line -/
noncomputable def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if a line passes through a point -/
def linePassesThroughPoint (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_proof (M N P : Point2D) (l : Line2D) 
    (h1 : M.x = 2 ∧ M.y = 3)
    (h2 : N.x = 4 ∧ N.y = -5)
    (h3 : P.x = 1 ∧ P.y = 2)
    (h4 : linePassesThroughPoint l P)
    (h5 : distancePointToLine M l = distancePointToLine N l) :
  (l.a = 3 ∧ l.b = 2 ∧ l.c = -7) ∨ (l.a = 4 ∧ l.b = 1 ∧ l.c = -6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l201_20102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l201_20177

/-- Calculates the compound interest given principal, rate, time, and compounding frequency -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time) - principal

/-- Theorem stating that the compound interest on $1200 for 3 years at 20% per annum, compounded yearly, is approximately $873.6 -/
theorem compound_interest_example : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |compound_interest 1200 0.20 3 1 - 873.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_example_l201_20177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_semicircle_l201_20118

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, 1 + 2 * Real.sin θ)

-- Define the range of θ
def θ_range (θ : ℝ) : Prop :=
  -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2

-- Theorem statement
theorem curve_is_semicircle :
  ∀ (x y : ℝ),
  (∃ θ, θ_range θ ∧ C θ = (x, y)) →
  (x^2 + (y - 1)^2 = 4 ∧ x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_semicircle_l201_20118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_and_phase_shift_l201_20131

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := sin (3 * x + π / 4)

-- State the theorem
theorem sin_period_and_phase_shift :
  -- The period is 2π/3
  (∀ x, f (x + 2*π/3) = f x) ∧ 
  -- The phase shift is -π/12
  (∀ x, f (x - π/12) = sin x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_period_and_phase_shift_l201_20131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l201_20146

-- Define the function f with domain (0,2)
noncomputable def f : {x : ℝ | 0 < x ∧ x < 2} → ℝ := sorry

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  if h : 0 < x - 4 ∧ x - 4 < 2 ∧ x > 5
  then f ⟨x - 4, h.1, h.2.1⟩ / Real.sqrt (x - 5)
  else 0

-- Theorem stating the domain of g
theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = {x : ℝ | 5 < x ∧ x < 6} := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l201_20146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_70_l201_20100

/-- The decimal representation of 17/70 has a repeating block of length 6 -/
def decimal_rep_17_70_period : ℕ := 6

/-- The repeating block in the decimal representation of 17/70 -/
def decimal_rep_17_70_block : List ℕ := [2, 4, 2, 8, 5, 7]

/-- The 150th digit in the decimal representation of 17/70 -/
def digit_150_of_17_70 : ℕ := 7

theorem digit_150_of_17_over_70 :
  digit_150_of_17_70 = (decimal_rep_17_70_block.get! ((150 - 1) % decimal_rep_17_70_period)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_17_over_70_l201_20100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_dividing_subset_size_l201_20116

theorem max_non_dividing_subset_size : ∃ (S : Finset ℕ), 
  (∀ x, x ∈ S → x ≤ 2020) ∧ 
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → ¬(x ∣ y)) ∧
  S.card = 1010 ∧
  (∀ T : Finset ℕ, (∀ x, x ∈ T → x ≤ 2020) → 
    (∀ x y, x ∈ T → y ∈ T → x ≠ y → ¬(x ∣ y)) → T.card ≤ 1010) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_dividing_subset_size_l201_20116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_lace_cost_l201_20149

/-- Represents the cost of lace for a dress with given measurements and lace prices. -/
noncomputable def laceCost (cuffLength waistLength hemLength necklineRuffles : ℝ) 
             (laceAPricePerMeter laceBPricePerMeter laceCPricePerMeter : ℝ) : ℝ :=
  let cuffTotal := 2 * cuffLength / 100
  let hemTotal := hemLength / 100
  let waistTotal := waistLength / 100
  let necklineTotal := 5 * necklineRuffles / 100
  let laceACost := (cuffTotal + hemTotal) * laceAPricePerMeter
  let laceBCost := waistTotal * laceBPricePerMeter
  let laceCCost := necklineTotal * laceCPricePerMeter
  laceACost + laceBCost + laceCCost

/-- Theorem stating that the total cost of lace for the dress is $44. -/
theorem dress_lace_cost :
  laceCost 50 100 300 20 6 8 12 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_lace_cost_l201_20149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_spending_calculation_l201_20148

noncomputable def weekly_pay : ℝ := 100
noncomputable def arcade_spending_ratio : ℝ := 1 / 2
noncomputable def play_rate_per_hour : ℝ := 8
noncomputable def play_time_minutes : ℝ := 300

noncomputable def arcade_spending : ℝ := weekly_pay * arcade_spending_ratio
noncomputable def play_time_hours : ℝ := play_time_minutes / 60
noncomputable def token_spending : ℝ := play_time_hours * play_rate_per_hour

theorem food_spending_calculation : 
  arcade_spending - token_spending = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_spending_calculation_l201_20148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l201_20168

theorem triangle_property (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  (a + c) / b = (Real.sin A - Real.sin B) / (Real.sin A - Real.sin C) →
  C = π / 3 ∧ 1 < (a + b) / c ∧ (a + b) / c < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l201_20168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l201_20189

-- Define the units of measurement
inductive MeasurementUnit
| Milliliter
| Ton
| Liter

-- Define the types of containers
inductive Container
| CoughSyrupBottle
| Warehouse
| GasolineBarrel

-- Function to determine the appropriate unit for a given container and capacity
def appropriate_unit (container : Container) (capacity : ℕ) : MeasurementUnit :=
  match container with
  | Container.CoughSyrupBottle => MeasurementUnit.Milliliter
  | Container.Warehouse => MeasurementUnit.Ton
  | Container.GasolineBarrel => MeasurementUnit.Liter

-- Theorem stating the appropriate units for each container
theorem appropriate_units :
  (appropriate_unit Container.CoughSyrupBottle 150 = MeasurementUnit.Milliliter) ∧
  (appropriate_unit Container.Warehouse 400 = MeasurementUnit.Ton) ∧
  (appropriate_unit Container.GasolineBarrel 150 = MeasurementUnit.Liter) := by
  sorry

#check appropriate_units

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l201_20189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l201_20121

theorem exponential_inequality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) : (2 : ℝ)^a < (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l201_20121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_binomial_coefficient_modulo_prime_l201_20107

theorem binomial_coefficient_divisibility (n k : ℕ) (h1 : k < n) (h2 : Nat.Coprime k n) :
  n ∣ Nat.choose n k := by
  sorry

theorem binomial_coefficient_modulo_prime (p : ℕ) (h : Nat.Prime p) :
  Nat.choose (2 * p) p ≡ 2 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_binomial_coefficient_modulo_prime_l201_20107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_2kpi_l201_20128

-- Define the function f(x) = 2cos(x) - 1
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x - 1

-- State the theorem
theorem f_max_at_2kpi :
  ∀ x : ℝ, ∃ k : ℤ, f (2 * π * ↑k) ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_2kpi_l201_20128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_l201_20143

theorem triangle_cosine_sum_max (A B C : ℝ) (h : A + B + C = Real.pi) : 
  Real.cos A + Real.cos B * Real.cos C ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_sum_max_l201_20143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_ten_factorial_l201_20197

theorem unique_divisor_ten_factorial : 
  ∃! d : ℕ, d ∣ Nat.factorial 10 ∧ d = 10 * Nat.factorial 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_ten_factorial_l201_20197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l201_20139

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1

/-- The coordinates of a focus of the hyperbola -/
noncomputable def focus : ℝ × ℝ := (Real.sqrt 13, 0)

/-- Theorem: The coordinates of the foci of the hyperbola x^2/4 - y^2/9 = 1 are (±√13, 0) -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola x y → (x, y) = focus ∨ (x, y) = (-focus.1, focus.2) := by
  sorry

#check hyperbola_foci

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l201_20139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_perfect_square_l201_20191

/-- A function representing the spiral sequence -/
def spiral (n : ℕ) : ℕ × ℕ := sorry

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- Predicate to check if a number is on the diagonal -/
def on_diagonal (n : ℕ) : Prop := is_perfect_square n

/-- Function to get the number above a given number in the spiral -/
def number_above (n : ℕ) : ℕ := sorry

theorem number_above_perfect_square (n : ℕ) (h : Even n) :
  on_diagonal (n^2) →
  number_above (n^2) = (n+2)^2 - 1 := by
  sorry

#check number_above_perfect_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_perfect_square_l201_20191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_S_l201_20174

noncomputable def S (a b c d : ℝ) : ℝ := 
  (a / (b + 7)) ^ (1/3) + (b / (c + 7)) ^ (1/3) + (c / (d + 7)) ^ (1/3) + (d / (a + 7)) ^ (1/3)

theorem max_value_S :
  ∀ a b c d : ℝ,
    a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
    a + b + c + d = 100 →
    S a b c d ≤ 4 * (25/32) ^ (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_S_l201_20174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_unit_circle_l201_20150

/-- The chord length cut by the unit circle from the line x = 1/2 is √3 -/
theorem chord_length_unit_circle : 
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x = 1/2 ∧ 
  2 * Real.sqrt (1 - (1/2)^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_unit_circle_l201_20150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_35_l201_20129

/-- The angle of the minute hand at 35 minutes past the hour -/
noncomputable def minute_hand_angle : ℝ := 35 / 60 * 360

/-- The angle of the hour hand at 7:35 -/
noncomputable def hour_hand_angle : ℝ := (7 + 35 / 60) / 12 * 360

/-- The acute angle between the hour hand and minute hand at 7:35 -/
noncomputable def angle_between_hands : ℝ := |hour_hand_angle - minute_hand_angle|

theorem clock_angle_at_7_35 : 
  ⌊angle_between_hands⌋ = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_35_l201_20129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_by_minute_hand_l201_20192

/-- The length of the minute hand in centimeters -/
noncomputable def minute_hand_length : ℝ := 15

/-- The area swept by the minute hand of a circular clock face in half an hour -/
noncomputable def area_swept (r : ℝ) : ℝ := (1/2) * Real.pi * r^2

/-- Theorem: The area swept by the minute hand of a circular clock face with a 15 cm long minute hand in half an hour is equal to (1/2) * π * 15^2 square centimeters -/
theorem area_swept_by_minute_hand :
  area_swept minute_hand_length = (1/2) * Real.pi * 15^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_swept_by_minute_hand_l201_20192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_passes_through_point_point_on_circle_triangle_area_l201_20141

-- Define the hyperbola
def Hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (2 * Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (-2 * Real.sqrt 3, 0)

-- Define the eccentricity
noncomputable def e : ℝ := Real.sqrt 2

-- Define a point on the hyperbola
def M (m : ℝ) : ℝ × ℝ := (3, m)

-- Theorem 1: The hyperbola passes through (4, -√10)
theorem hyperbola_passes_through_point :
  Hyperbola 4 (-Real.sqrt 10) := by sorry

-- Theorem 2: Any point M(3, m) on the hyperbola lies on the circle with F₁F₂ as diameter
theorem point_on_circle (m : ℝ) (h : Hyperbola 3 m) :
  let M := (3, m)
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0 := by sorry

-- Theorem 3: The area of triangle F₁MF₂ is 6
theorem triangle_area (m : ℝ) (h : Hyperbola 3 m) :
  let M := (3, m)
  let area := Real.sqrt (
    ((F₁.1 - M.1)^2 + (F₁.2 - M.2)^2) *
    ((F₂.1 - M.1)^2 + (F₂.2 - M.2)^2) *
    ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)
  ) / 4
  area = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_passes_through_point_point_on_circle_triangle_area_l201_20141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_with_zero_discriminant_l201_20161

theorem quadratic_roots_with_zero_discriminant :
  ∀ (p : ℝ),
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 - 4 * p * x + 9
  let discriminant := (-4 * p)^2 - 4 * 3 * 9
  discriminant = 0 →
  ∃ (x : ℝ), f x = 0 ∧ (x = Real.sqrt 3 ∨ x = -Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_with_zero_discriminant_l201_20161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l201_20103

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (3, 5)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 3*x - 4*y + 11 = 0
def tangent_line_2 (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem tangent_lines_to_circle :
  (∀ x y, tangent_line_1 x y → (x, y) = point_P ∨ (∃ t, my_circle t y ∧ (x, y) = (t, y))) ∧
  (∀ x y, tangent_line_2 x → (x, y) = point_P ∨ (∃ t, my_circle x t ∧ (x, y) = (x, t))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l201_20103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_one_l201_20176

-- Define the function f(x) = √(x-1)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1)

-- Define the domain of f
def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

-- Theorem stating that the domain of f is {x ∈ ℝ | x ≥ 1}
theorem domain_of_sqrt_x_minus_one :
  domain f = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_x_minus_one_l201_20176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_parallel_line_equation_l201_20187

-- Define the parabola
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (1/2, 0)

-- Define the parallel line
noncomputable def parallel_line (x y : ℝ) : Prop := 3*x - 2*y + 5 = 0

-- Define the line passing through the focus and parallel to the given line
noncomputable def focus_parallel_line (x y : ℝ) : Prop := 6*x - 4*y - 3 = 0

-- Theorem statement
theorem focus_parallel_line_equation : 
  ∀ (x y : ℝ), focus_parallel_line x y ↔ 
  (∃ (k : ℝ), parallel_line (x + k) (y + (3/2)*k)) ∧ 
  (x = focus.1 ∧ y = focus.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_parallel_line_equation_l201_20187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l201_20124

/-- Prove that the speed of the second train is 80.016 kmph given the following conditions:
  1. First train length: 200 meters
  2. First train speed: 120 kmph
  3. Time to cross: 9 seconds
  4. Second train length: 300.04 meters
-/
theorem second_train_speed 
  (first_train_length : ℝ) 
  (first_train_speed : ℝ) 
  (crossing_time : ℝ) 
  (second_train_length : ℝ)
  (h1 : first_train_length = 200)
  (h2 : first_train_speed = 120)
  (h3 : crossing_time = 9)
  (h4 : second_train_length = 300.04) :
  let total_length := first_train_length + second_train_length
  let total_length_km := total_length / 1000
  let crossing_time_hours := crossing_time / 3600
  let relative_speed := total_length_km / crossing_time_hours
  let second_train_speed := relative_speed - first_train_speed
  second_train_speed = 80.016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l201_20124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l201_20144

open Real BigOperators

theorem infinite_sum_convergence : 
  (∑' k : ℕ, (8^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l201_20144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_l201_20147

def count_quadruples (M N : ℕ) : ℕ :=
  (Finset.range N).card^4

theorem quadruple_count (M N : ℕ) (h : M ≤ N) :
  (Finset.filter 
    (λ (t : ℕ × ℕ × ℕ × ℕ) => 
      let (w, x, y, z) := t
      w + x + y + z = M * Real.sqrt (↑(w * x * y * z)))
    (Finset.product 
      (Finset.range N) 
      (Finset.product 
        (Finset.range N) 
        (Finset.product 
          (Finset.range N) 
          (Finset.range N))))).card
  = count_quadruples M N :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_count_l201_20147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l201_20110

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  side_positive : a > 0 ∧ b > 0 ∧ c > 0

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h : Real.sin (t.A - t.B) * Real.cos t.C = Real.cos t.B * Real.sin (t.A - t.C)) :
  -- Part 1: Triangle is right or isosceles
  (t.A = Real.pi / 2 ∨ t.B = t.C) ∧
  -- Part 2: Maximum value for acute triangle
  (∀ (acute : t.A < Real.pi / 2 ∧ t.B < Real.pi / 2 ∧ t.C < Real.pi / 2) 
      (side_a : t.a = 1 / Real.sin t.B),
    ∃ (max : Real), max = 25 / 16 ∧
    ∀ (x : Real), (t.b^2 + t.a^2) / (t.a * t.b)^2 ≤ max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l201_20110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_no_maximum_l201_20159

/-- A parabola with a specific tangent property -/
structure Parabola where
  b : ℝ
  c : ℝ
  tangent_point : ℝ × ℝ := (1, 2)
  tangent_perpendicular : (∀ x y : ℝ, x + y + 2 = 0 → (2 + b) * (-1) = -1)

/-- The function represented by the parabola -/
def parabola_function (p : Parabola) (x : ℝ) : ℝ :=
  x^2 + p.b * x + p.c

theorem parabola_minimum_no_maximum (p : Parabola) :
  (∃ x : ℝ, parabola_function p x = 7/4 ∧ ∀ y : ℝ, parabola_function p y ≥ 7/4) ∧
  (¬∃ M : ℝ, ∀ x : ℝ, parabola_function p x ≤ M) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_no_maximum_l201_20159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l201_20132

-- Define the line l: x - y + m = 0
def line (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

-- Define the circle (x-3)^2 + (y-2)^2 = 6
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 6

-- Define the center of the circle
def center : ℝ × ℝ := (3, 2)

-- Define the property of line l intersecting the circle at points P and Q
def intersects (m : ℝ) : Prop := ∃ (p q : ℝ × ℝ), p ≠ q ∧ 
  line m p.1 p.2 ∧ line m q.1 q.2 ∧ 
  circle_eq p.1 p.2 ∧ circle_eq q.1 q.2

-- Define the property of triangle MPQ being equilateral
def is_equilateral (m : ℝ) : Prop := 
  ∃ (p q : ℝ × ℝ), p ≠ q ∧
  line m p.1 p.2 ∧ line m q.1 q.2 ∧ 
  circle_eq p.1 p.2 ∧ circle_eq q.1 q.2 ∧
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = 
  (q.1 - center.1)^2 + (q.2 - center.2)^2 ∧
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = 
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Theorem statement
theorem line_circle_intersection (m : ℝ) : 
  intersects m → is_equilateral m → m = 2 ∨ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l201_20132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersect_property_l201_20184

/-- Given a function f(x) = tan(ωx) where ω > 0, if the distance between two adjacent
    intersections of y = π/4 with f(x) is π/4, then f(π/4) = 0. -/
theorem tan_intersect_property (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ x : ℝ, Real.tan (ω * x) = π / 4 ∧ Real.tan (ω * (x + π / 4)) = π / 4) →
  Real.tan (ω * (π / 4)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_intersect_property_l201_20184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_12_l201_20163

/-- Represents the swimming scenario --/
structure SwimmingScenario where
  stillWaterSpeed : ℚ
  downstreamDistance : ℚ
  swimmingTime : ℚ

/-- Calculates the stream speed given a swimming scenario --/
def calculateStreamSpeed (s : SwimmingScenario) : ℚ :=
  (s.downstreamDistance / s.swimmingTime - s.stillWaterSpeed) / 2

/-- Calculates the upstream distance given a swimming scenario --/
def calculateUpstreamDistance (s : SwimmingScenario) : ℚ :=
  (s.stillWaterSpeed - calculateStreamSpeed s) * s.swimmingTime

/-- Theorem stating the upstream distance for the given scenario --/
theorem upstream_distance_is_12 (s : SwimmingScenario) 
  (h1 : s.stillWaterSpeed = 5)
  (h2 : s.downstreamDistance = 18)
  (h3 : s.swimmingTime = 3) :
  calculateUpstreamDistance s = 12 := by
  sorry

#eval calculateUpstreamDistance { stillWaterSpeed := 5, downstreamDistance := 18, swimmingTime := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_12_l201_20163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l201_20157

-- Define the points
variable (X Y Z L M N O P Q : ℝ × ℝ)

-- Define the equilateral triangle XYZ
def is_equilateral (X Y Z : ℝ × ℝ) : Prop :=
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

-- Define parallel line segments
def are_parallel (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.2 - C.2) = (D.1 - C.1) * (B.2 - A.2)

-- Define the condition that XL = LN = NP = PY
def equal_segments (X L N P Y : ℝ × ℝ) : Prop :=
  dist X L = dist L N ∧ dist L N = dist N P ∧ dist N P = dist P Y

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Define the area of a trapezoid
noncomputable def trapezoid_area (A B C D : ℝ × ℝ) : ℝ :=
  abs ((A.1 - D.1) * (B.2 - A.2) + (B.1 - A.1) * (C.2 - B.2) + 
       (C.1 - B.1) * (D.2 - C.2)) / 2

-- State the theorem
theorem area_ratio_theorem (X Y Z L M N O P Q : ℝ × ℝ) :
  is_equilateral X Y Z →
  are_parallel L M Y Z →
  are_parallel N O Y Z →
  are_parallel P Q Y Z →
  equal_segments X L N P Y →
  trapezoid_area P Q Y Z / triangle_area X Y Z = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l201_20157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l201_20179

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_properties :
  let tangent_line (x : ℝ) := x - 1
  ∃ (a : ℝ),
    (∀ x, x > 0 → (deriv f x = Real.log x + 1)) ∧
    (StrictAntiOn f (Set.Ioo 0 (Real.exp (-1)))) ∧
    (StrictMonoOn f (Set.Ioi (Real.exp (-1)))) ∧
    (∀ x, x ∈ Set.Icc (Real.exp (-1)) (Real.exp 1) → f x ≤ a * x - 1) ∧
    (a = Real.exp 1 - 1) ∧
    (tangent_line 1 = f 1) ∧
    (HasDerivAt f (tangent_line 1 - f 1) 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l201_20179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_implies_m_l201_20130

/-- A line in the Cartesian coordinate system -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

/-- The problem statement -/
theorem y_intercept_implies_m (m : ℝ) :
  let l : Line := { a := 1, b := -2, c := m - 1 }
  y_intercept l = 1/2 → m = 2 := by
  sorry

#check y_intercept_implies_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_implies_m_l201_20130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_rule_accuracy_l201_20137

-- Define the function to be integrated
noncomputable def f (x : ℝ) := x^2

-- Define the integral bounds
def a : ℝ := 0
def b : ℝ := 4

-- Define the number of subdivisions
def n : ℕ := 10

-- Define the width of each subdivision
noncomputable def Δx : ℝ := (b - a) / n

-- Define the trapezoidal rule approximation
noncomputable def trapezoidal_approx : ℝ :=
  (Δx / 2) * (f a + f b + 2 * (Finset.sum (Finset.range (n - 1)) (λ i => f (a + (i + 1) * Δx))))

-- Define the exact value of the integral
noncomputable def exact_integral : ℝ := (b^3 - a^3) / 3

-- State the theorem
theorem trapezoidal_rule_accuracy :
  abs ((trapezoidal_approx - exact_integral) / exact_integral) ≤ 0.005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_rule_accuracy_l201_20137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buratino_apple_game_l201_20123

/-- Given six distinct real numbers, there exist four real numbers such that
    their pairwise sums matched with the given numbers result in at least
    four equalities and two inequalities (where the sum is greater),
    and this is the maximum guaranteed result. -/
theorem buratino_apple_game (a b c d e f : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                                               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                                               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                                               d ≠ e ∧ d ≠ f ∧
                                               e ≠ f) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ),
    (∃ (p : Equiv.Perm (Fin 6)),
      (x₁ + x₂ = Fin.val (p 0)) ∧
      (x₁ + x₃ = Fin.val (p 1)) ∧
      (x₂ + x₃ = Fin.val (p 2)) ∧
      (x₃ + x₄ = Fin.val (p 3)) ∧
      (x₁ + x₄ > Fin.val (p 4)) ∧
      (x₂ + x₄ > Fin.val (p 5))) ∧
    (∀ (y₁ y₂ y₃ y₄ : ℝ),
      ¬∃ (q : Equiv.Perm (Fin 6)),
        ((y₁ + y₂ = Fin.val (q 0)) ∧
         (y₁ + y₃ = Fin.val (q 1)) ∧
         (y₁ + y₄ = Fin.val (q 2)) ∧
         (y₂ + y₃ = Fin.val (q 3)) ∧
         (y₂ + y₄ = Fin.val (q 4)) ∧
         (y₃ + y₄ = Fin.val (q 5)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_buratino_apple_game_l201_20123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_formula_l201_20182

def sequence_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) a

def sequence_a (s : ℕ → ℚ) (n : ℕ) : ℚ :=
  if n = 1 then -2/3 else s n + 1 / s n + 2

theorem sequence_sum_formula (s : ℕ → ℚ) :
  (∀ n : ℕ, n ≥ 2 → s n + 1 / s n + 2 = sequence_a s n) →
  s 1 = -2/3 →
  ∀ n : ℕ, s n = -(n + 1 : ℚ) / (n + 2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_formula_l201_20182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_days_for_all_meetings_l201_20104

/-- Represents the number of friends --/
def num_friends : ℕ := 45

/-- Represents the total number of proposed days --/
def total_days : ℕ := 10

/-- Represents the number of days each friend can attend --/
def attendance_days : ℕ := 8

/-- 
Theorem: Given 45 friends, each able to attend 8 out of 10 proposed days, 
the minimum number of days needed to ensure every pair of friends 
meet at least once is 5.
-/
theorem min_days_for_all_meetings : 
  ∃ (k : ℕ), k = 5 ∧ 
  (∀ (days : Finset ℕ), days.card = k → days ⊆ Finset.range total_days →
    ∀ (i j : Fin num_friends), i ≠ j → 
      ∃ (d : ℕ), d ∈ days ∧ 
        d ∈ (Finset.range total_days).filter (λ x ↦ x ∈ (Finset.range total_days).filter (λ y ↦ y ∉ (Finset.range total_days).filter (λ z ↦ z ∉ (Finset.range total_days).filter (λ w ↦ w ∈ (Finset.range total_days))))) ∧
        d ∈ (Finset.range total_days).filter (λ x ↦ x ∈ (Finset.range total_days).filter (λ y ↦ y ∉ (Finset.range total_days).filter (λ z ↦ z ∉ (Finset.range total_days).filter (λ w ↦ w ∈ (Finset.range total_days)))))) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (days : Finset ℕ), days.card = k' ∧ days ⊆ Finset.range total_days ∧
      ∃ (i j : Fin num_friends), i ≠ j ∧ 
        ∀ (d : ℕ), d ∈ days → 
          d ∉ (Finset.range total_days).filter (λ x ↦ x ∈ (Finset.range total_days).filter (λ y ↦ y ∉ (Finset.range total_days).filter (λ z ↦ z ∉ (Finset.range total_days).filter (λ w ↦ w ∈ (Finset.range total_days))))) ∨
          d ∉ (Finset.range total_days).filter (λ x ↦ x ∈ (Finset.range total_days).filter (λ y ↦ y ∉ (Finset.range total_days).filter (λ z ↦ z ∉ (Finset.range total_days).filter (λ w ↦ w ∈ (Finset.range total_days))))))
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_days_for_all_meetings_l201_20104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skip_speed_ratio_l201_20108

/-- The time taken for a single skip in seconds -/
def single_skip_time : ℝ := 0.5

/-- The time taken for a double skip in seconds -/
def double_skip_time : ℝ := 0.6

/-- The number of rotations in a single skip -/
def single_skip_rotations : ℕ := 1

/-- The number of rotations in a double skip -/
def double_skip_rotations : ℕ := 2

/-- The radius of the circle traced by the midpoint of the rope -/
noncomputable def r : ℝ := Real.pi  -- Using pi as a placeholder value

/-- The speed of the midpoint during a single skip -/
noncomputable def single_skip_speed : ℝ := (2 * Real.pi * r * single_skip_rotations) / single_skip_time

/-- The speed of the midpoint during a double skip -/
noncomputable def double_skip_speed : ℝ := (2 * Real.pi * r * double_skip_rotations) / double_skip_time

theorem skip_speed_ratio :
  single_skip_speed / double_skip_speed = 3 / 5 := by
  -- Expand definitions
  unfold single_skip_speed double_skip_speed
  -- Simplify fractions
  simp [single_skip_time, double_skip_time, single_skip_rotations, double_skip_rotations]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skip_speed_ratio_l201_20108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_theorem_l201_20198

noncomputable def A : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def B : ℝ × ℝ := (Real.sqrt 2, 0)

def C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

def midpoint_condition (x y : ℝ) : Prop := x + 2 * y = 0

theorem trajectory_and_line_theorem :
  -- Part 1: Trajectory equation
  (∀ x y : ℝ, x ≠ -Real.sqrt 2 → x ≠ Real.sqrt 2 →
    (y / (x + Real.sqrt 2)) * (y / (x - Real.sqrt 2)) = -1/2 → C x y) ∧
  -- Part 2: Line equation
  (∃ k : ℝ, ∀ x₁ y₁ x₂ y₂ x₀ y₀ : ℝ,
    C x₁ y₁ → C x₂ y₂ → l k x₁ y₁ → l k x₂ y₂ →
    x₀ = (x₁ + x₂) / 2 → y₀ = (y₁ + y₂) / 2 →
    midpoint_condition x₀ y₀ → k = 1) := by
  sorry

#check trajectory_and_line_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_line_theorem_l201_20198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slowest_sheep_eats_one_bag_in_40_days_l201_20119

/-- The number of days it takes for the slowest sheep to eat one bag of grass -/
noncomputable def slowest_sheep_days (num_sheep : ℕ) (num_bags : ℕ) (total_days : ℕ) (fastest_rate_multiplier : ℕ) : ℕ :=
  let avg_consumption_per_day := (num_bags : ℚ) / total_days
  let avg_consumption_per_sheep_per_day := avg_consumption_per_day / num_sheep
  let slowest_rate := 2 * avg_consumption_per_sheep_per_day / (1 + fastest_rate_multiplier)
  ⌈(1 / slowest_rate)⌉.toNat

/-- 
Given that 30 sheep eat 45 bags of grass in 40 days, and the fastest sheep eats at twice the rate 
of the slowest sheep, prove that one slowest sheep will eat one bag of grass in 40 days.
-/
theorem slowest_sheep_eats_one_bag_in_40_days :
  slowest_sheep_days 30 45 40 2 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slowest_sheep_eats_one_bag_in_40_days_l201_20119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_distance_theorem_l201_20151

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Defines a dilation transformation -/
def dilation (center : Point) (k : ℝ) (p : Point) : Point :=
  { x := center.x + k * (p.x - center.x)
  , y := center.y + k * (p.y - center.y) }

theorem dilation_distance_theorem :
  let original_circle : Circle := { center := { x := 3, y := 3 }, radius := 4 }
  let dilated_circle : Circle := { center := { x := 7, y := 9 }, radius := 6 }
  let point_p : Point := { x := 1, y := 1 }
  let k : ℝ := dilated_circle.radius / original_circle.radius
  let dilation_center : Point := 
    { x := (3*7 - 4*3) / (3 - 4), y := (3*9 - 4*3) / (3 - 4) }
  distance point_p (dilation dilation_center k point_p) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_distance_theorem_l201_20151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_l201_20181

noncomputable def angle_at_vertex_of_axial_section_of_cone (α : Real) : Real :=
  2 * Real.arcsin (α / (2 * Real.pi))

theorem cone_vertex_angle (α : Real) (h1 : α > 0) (h2 : α < 2 * Real.pi) : 
  angle_at_vertex_of_axial_section_of_cone α = 2 * Real.arcsin (α / (2 * Real.pi)) :=
by
  unfold angle_at_vertex_of_axial_section_of_cone
  rfl

#check cone_vertex_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_vertex_angle_l201_20181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l201_20140

/-- Calculates the speed of a train in km/hr given its length, platform length, and time to cross -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / time
  3.6 * speed_ms

/-- Theorem stating that a train with given parameters has a specific speed -/
theorem train_speed_calculation :
  let train_length := (250 : ℝ)
  let platform_length := (300 : ℝ)
  let time := (35.99712023038157 : ℝ)
  abs (train_speed train_length platform_length time - 54.9996) < 0.0001 := by
  sorry

-- Use #eval only for computable functions
def approximate_train_speed : ℚ :=
  let train_length : ℚ := 250
  let platform_length : ℚ := 300
  let time : ℚ := 35997120230381570 / 1000000000000000
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / time
  (36 * speed_ms) / 10

#eval approximate_train_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l201_20140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_eq_const_l201_20195

/-- A pentagon formed by placing an equilateral triangle on top of a square -/
structure TriangleSquarePentagon where
  side : ℝ
  side_pos : side > 0

namespace TriangleSquarePentagon

/-- The area of the square part of the pentagon -/
def square_area (p : TriangleSquarePentagon) : ℝ := p.side ^ 2

/-- The area of the equilateral triangle part of the pentagon -/
noncomputable def triangle_area (p : TriangleSquarePentagon) : ℝ := (Real.sqrt 3 / 4) * p.side ^ 2

/-- The total area of the pentagon -/
noncomputable def total_area (p : TriangleSquarePentagon) : ℝ := p.square_area + p.triangle_area

/-- The fraction of the pentagon's area occupied by the equilateral triangle -/
noncomputable def triangle_fraction (p : TriangleSquarePentagon) : ℝ := p.triangle_area / p.total_area

theorem triangle_fraction_eq_const (p : TriangleSquarePentagon) :
  p.triangle_fraction = Real.sqrt 3 / (4 + Real.sqrt 3) := by
  sorry

end TriangleSquarePentagon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_eq_const_l201_20195
