import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cube_plus_sin_l225_22500

theorem min_value_sin_cube_plus_sin : 
  ∃ (m : ℝ), m = -3 ∧ ∀ x : ℝ, -Real.sin x^3 - 2 * Real.sin x ≥ m :=
by
  -- We'll use m = -3 as the minimum value
  use -3
  constructor
  · -- Prove m = -3
    rfl
  · -- Prove ∀ x : ℝ, -Real.sin x^3 - 2 * Real.sin x ≥ -3
    intro x
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cube_plus_sin_l225_22500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_profit_disks_l225_22581

/-- The number of disks Maria needs to sell to make a profit of $200 -/
def disks_to_sell : ℕ :=
  let buy_price : ℚ := 10 / 6
  let sell_price : ℚ := 10 / 5
  let profit_per_disk : ℚ := sell_price - buy_price
  let disks_needed : ℚ := 200 / profit_per_disk
  (disks_needed.ceil.toNat)

/-- Proof that Maria needs to sell 607 disks to make a profit of $200 -/
theorem maria_profit_disks : disks_to_sell = 607 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_profit_disks_l225_22581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_initial_value_l225_22546

def compute_sum (x : Int) : Int :=
  let rec loop (S : Int) (a : Int) (I : List Int) : Int :=
    match I with
    | [] => S
    | i :: rest => loop (S + a * i) (a * (-1)) rest

  loop 0 x [1, 3, 5, 7, 9]

theorem correct_initial_value :
  compute_sum (-1) = -1 + 3 - 5 + 7 - 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_initial_value_l225_22546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_largest_divisor_l225_22565

def n : ℕ := 645120000

-- Define a function to get the list of divisors
def divisors (m : ℕ) : List ℕ :=
  (List.range (m + 1)).filter (λ k => m % k = 0)

-- Define a function to get the nth largest element of a list
def nthLargest (l : List ℕ) (k : ℕ) : ℕ :=
  (l.reverse.nthLe (k - 1) sorry)

theorem fifth_largest_divisor :
  nthLargest (divisors n) 5 = 40320000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_largest_divisor_l225_22565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_product_of_distances_l225_22576

noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

def line_l (x y : ℝ) : Prop := x - y - 6 = 0

def point_M : ℝ × ℝ := (-1, 0)

def distance_to_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry

theorem max_distance_point :
  ∃ (α : ℝ), curve_C α = (-3/2, 1/2) ∧
  ∀ (β : ℝ), distance_to_line (curve_C β) line_l ≤ distance_to_line (curve_C α) line_l ∧
  distance_to_line (curve_C α) line_l = 4 * Real.sqrt 2 :=
by sorry

def line_l1 (x y : ℝ) : Prop :=
  ∃ (k : ℝ), x - y = k ∧ line_l (point_M.1 + x) (point_M.2 + y)

theorem product_of_distances :
  ∃ (A B : ℝ × ℝ), ∃ (α β : ℝ), 
    curve_C α = A ∧ curve_C β = B ∧
    line_l1 A.1 A.2 ∧ line_l1 B.1 B.2 ∧
    ((A.1 - point_M.1)^2 + (A.2 - point_M.2)^2) *
    ((B.1 - point_M.1)^2 + (B.2 - point_M.2)^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_product_of_distances_l225_22576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monochromatic_isosceles_triangles_l225_22583

-- Define a circle
def Circle : Type := ℝ × ℝ

-- Define a color type
inductive Color
| Red
| Green
| Blue

-- Define a point on the circle
structure Point where
  position : Circle
  color : Color

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : Point
  b : Point
  c : Point
  isIsosceles : True  -- We assume this property holds for simplicity

-- Define a coloring of the circle
def Coloring := Circle → Color

-- The main theorem
theorem infinite_monochromatic_isosceles_triangles 
  (coloring : Coloring) : 
  ∃ (triangles : Set IsoscelesTriangle), 
    (Set.Infinite triangles) ∧ 
    (∀ t ∈ triangles, t.a.color = t.b.color ∧ t.b.color = t.c.color) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monochromatic_isosceles_triangles_l225_22583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l225_22571

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (the focus) and a fixed line (the directrix). -/
structure Parabola where
  /-- The x-coordinate of the directrix -/
  directrix : ℝ

/-- The standard equation of a parabola is of the form y² = kx, where k is a non-zero constant. -/
def standardEquation (p : Parabola) : ℝ → Prop :=
  fun k => ∀ x y, y^2 = k * x ↔ (x - p.directrix)^2 = (x - (p.directrix - 1/2))^2 + y^2

/-- Theorem: For a parabola with directrix x = 1, its standard equation is y² = -8x -/
theorem parabola_standard_equation (p : Parabola) (h : p.directrix = 1) :
  standardEquation p (-8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_standard_equation_l225_22571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l225_22597

open BigOperators

def f (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, 1 / (k + n + 1)

theorem f_difference (n : ℕ) : 
  f (n + 1) - f n = 1 / (2*n + 1) - 1 / (2*n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l225_22597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_a_range_l225_22589

/-- The minimum value of a for which there exists a common tangent line between y=ax^2 and y=e^x -/
noncomputable def min_a : ℝ := Real.exp 2 / 4

theorem common_tangent_a_range (a : ℝ) :
  (a > 0) →
  (∃ (x₁ x₂ : ℝ), (2 * a * x₁ = Real.exp x₂) ∧ 
                   (2 * a * x₁ = (Real.exp x₂ - a * x₁^2) / (x₂ - x₁))) →
  a ≥ min_a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_a_range_l225_22589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_problem_l225_22506

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  nextInningsRuns : ℕ
  averageIncrease : ℕ

/-- Calculates the current average of runs per innings -/
def currentAverage (player : CricketPlayer) : ℚ :=
  player.totalRuns / player.innings

/-- Calculates the new average after playing the next innings -/
def newAverage (player : CricketPlayer) : ℚ :=
  (player.totalRuns + player.nextInningsRuns) / (player.innings + 1)

/-- Theorem: Given the conditions, prove that the current average is 32 -/
theorem cricket_average_problem (player : CricketPlayer)
  (h1 : player.innings = 10)
  (h2 : player.nextInningsRuns = 76)
  (h3 : newAverage player - currentAverage player = player.averageIncrease)
  (h4 : player.averageIncrease = 4) :
  currentAverage player = 32 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_problem_l225_22506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_shared_correct_l225_22596

/-- Represents the number of apples a person has -/
structure Apples where
  count : Nat

/-- Craig's initial number of apples -/
def craig_initial : Apples := ⟨20⟩

/-- Craig's number of apples after sharing -/
def craig_after : Apples := ⟨13⟩

/-- The number of apples Craig shared with Eugene -/
def apples_shared : Nat := craig_initial.count - craig_after.count

theorem apples_shared_correct : apples_shared = 7 := by
  rfl

#eval apples_shared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_shared_correct_l225_22596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_pot_cost_approx_l225_22530

/-- The cost of the largest pot given a set of 6 pots with increasing prices -/
def largest_pot_cost (num_pots : ℕ) (total_cost : ℚ) (price_difference : ℚ) : ℚ :=
  let smallest_pot_cost := (total_cost - price_difference * (num_pots * (num_pots - 1) / 2)) / num_pots
  smallest_pot_cost + price_difference * (num_pots - 1)

/-- Theorem stating the cost of the largest pot -/
theorem largest_pot_cost_approx :
  ∃ (ε : ℚ), ε > 0 ∧ |largest_pot_cost 6 (825/100) (3/10) - 229/100| < ε := by
  sorry

#eval largest_pot_cost 6 (825/100) (3/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_pot_cost_approx_l225_22530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l225_22520

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l225_22520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_salary_after_tax_is_205_2_l225_22584

/-- The weekly salary of employee B after tax deduction -/
noncomputable def salary_B_after_tax (total_salary : ℝ) (a_to_b_ratio : ℝ) (tax_rate : ℝ) : ℝ :=
  let salary_B_before_tax := total_salary / (a_to_b_ratio + 1)
  salary_B_before_tax * (1 - tax_rate)

/-- Theorem stating that B's salary after tax is 205.2 given the problem conditions -/
theorem b_salary_after_tax_is_205_2 :
  salary_B_after_tax 570 1.5 0.1 = 205.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_salary_after_tax_is_205_2_l225_22584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_special_trigonometric_matrix_l225_22570

open Matrix Real

theorem determinant_special_trigonometric_matrix (a b : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![
    1, sin (a - b), sin a;
    sin (a - b), 1, sin b;
    sin a, sin b, 1
  ]
  det M = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_special_trigonometric_matrix_l225_22570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_equation_l225_22525

/-- Given a parabola y^2 = 8x, prove that the circle with its focus as the center 
    and the diameter formed by the intersection points of the parabola and a line 
    perpendicular to the x-axis through the focus has the equation (x-2)^2 + y^2 = 16 -/
theorem parabola_circle_equation (x y : ℝ) : 
  (∃ (a b : ℝ), y^2 = 8*x ∧ a^2 = 8*2 ∧ b^2 = 8*2 ∧ a ≠ b) → 
  ((x - 2)^2 + y^2 = 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_equation_l225_22525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_day_correct_l225_22540

/-- Represents the number of days it takes for the horse and donkey to meet -/
def meetingDay : ℕ := 9

/-- The distance between Chang'an and Qi in miles -/
def totalDistance : ℚ := 1125

/-- The horse's travel distance on the first day in miles -/
def horseInitialDistance : ℚ := 103

/-- The horse's daily increase in travel distance in miles -/
def horseDailyIncrease : ℚ := 13

/-- The donkey's travel distance on the first day in miles -/
def donkeyInitialDistance : ℚ := 97

/-- The donkey's daily decrease in travel distance in miles -/
def donkeyDailyDecrease : ℚ := 1/2

/-- The horse's travel distance on the nth day -/
def horseDistance (n : ℕ) : ℚ := horseInitialDistance + horseDailyIncrease * (n - 1)

/-- The donkey's travel distance on the nth day -/
def donkeyDistance (n : ℕ) : ℚ := donkeyInitialDistance - donkeyDailyDecrease * (n - 1)

/-- The total distance traveled by both animals after n days -/
noncomputable def totalTraveledDistance (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (horseInitialDistance + horseDistance n) +
  (n : ℚ) / 2 * (donkeyInitialDistance + donkeyDistance n)

theorem meeting_day_correct :
  totalTraveledDistance meetingDay = 2 * totalDistance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_day_correct_l225_22540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l225_22557

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1/2
  | n+1 => let a := sequence_a n; a^2 / (a^2 - 2*a + 2)

theorem sequence_inequality (n : ℕ) (h : n ≥ 2) :
  (Finset.range n).sum (λ i => (1/2:ℝ)^i * sequence_a (n-i-1)) < (1/2:ℝ)^(n-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l225_22557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l225_22516

noncomputable section

/-- Curve C in parametric form -/
def curve_C (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sin θ)

/-- Line l in parametric form -/
def line_l (a t : ℝ) : ℝ × ℝ := (a + 4 * t, 1 - t)

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
  let (x, y) := p
  abs (x + 4 * y - a - 4) / Real.sqrt 17

theorem intersection_points_and_max_distance :
  (∃ θ₁ θ₂ : ℝ, 
    curve_C θ₁ = line_l (-1) (1/4) ∧ 
    curve_C θ₂ = line_l (-1) (21/100)) ∧
  (∀ a : ℝ, 
    (∀ θ : ℝ, distance_point_to_line (curve_C θ) a ≤ Real.sqrt 17) →
    (∃ θ : ℝ, distance_point_to_line (curve_C θ) a = Real.sqrt 17) →
    (a = -16 ∨ a = 8)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_max_distance_l225_22516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l225_22504

-- Define the necessary structures and functions
def is_square (S : Set (ℝ × ℝ)) : Prop := sorry
def is_side (S : Set (ℝ × ℝ)) (side : Set (ℝ × ℝ)) : Prop := sorry
def is_vertex (S : Set (ℝ × ℝ)) (v : ℝ × ℝ) : Prop := sorry
def side_length (S : Set (ℝ × ℝ)) : ℝ := sorry
def extend (side : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry
def collinear (S : Set (ℝ × ℝ)) (v : ℝ × ℝ) : Prop := sorry
def distance (a b : ℝ × ℝ) : ℝ := sorry
def area (S : Set (ℝ × ℝ)) : ℝ := sorry

theorem inner_square_area (WXYZ IJKL : Set (ℝ × ℝ)) 
  (h1 : is_square WXYZ) 
  (h2 : is_square IJKL) 
  (h3 : IJKL ⊆ WXYZ) 
  (h4 : side_length WXYZ = Real.sqrt 120) 
  (h5 : ∀ (side : Set (ℝ × ℝ)), is_side IJKL side → 
    ∃ (v : ℝ × ℝ), is_vertex WXYZ v ∧ collinear (extend side) v)
  (h6 : distance (0, 0) (3, 3) = 3) :
  area IJKL = (Real.sqrt 111 - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l225_22504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_multiplication_l225_22519

/-- Represents a digit in base 6 -/
def Base6Digit := Fin 6

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List Base6Digit := sorry

/-- Converts a base 6 number to its decimal representation -/
def fromBase6 (digits : List Base6Digit) : ℕ := sorry

/-- Multiplies two numbers in base 6 -/
def multBase6 (a b : List Base6Digit) : List Base6Digit := sorry

theorem base6_multiplication :
  let a := [⟨3, sorry⟩, ⟨5, sorry⟩, ⟨2, sorry⟩]
  let b := [⟨3, sorry⟩, ⟨1, sorry⟩]
  let c := [⟨2, sorry⟩, ⟨0, sorry⟩, ⟨1, sorry⟩, ⟨5, sorry⟩, ⟨2, sorry⟩]
  multBase6 a b = c ∧ fromBase6 (multBase6 a b) = fromBase6 a * fromBase6 b := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base6_multiplication_l225_22519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_customers_count_l225_22593

theorem waiter_customers_count
  (num_tables : ℕ)
  (women_per_table : ℕ)
  (men_per_table : ℕ)
  (total_customers : ℕ) :
  num_tables = 6 →
  women_per_table = 3 →
  men_per_table = 5 →
  total_customers = num_tables * (women_per_table + men_per_table) →
  total_customers = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_customers_count_l225_22593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_line_property_l225_22508

-- Define a type for barycentric coordinates
def BarycentricCoord (n : ℕ) := Fin n → ℝ

-- Define a line in barycentric coordinates
structure BarycentricLine (n : ℕ) where
  contains : BarycentricCoord n → Prop

-- Define a point in barycentric coordinates
def BarycentricPoint (n : ℕ) := BarycentricCoord n

theorem barycentric_line_property {n : ℕ} (L : BarycentricLine n) 
  (x y : BarycentricCoord n) :
  L.contains x → L.contains y → 
  L.contains (λ i => x i + y i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barycentric_line_property_l225_22508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l225_22551

-- Define the function
noncomputable def f (x : ℝ) := Real.log (x^2 - x - 6) / Real.log (1/2)

-- State the theorem
theorem monotonic_increasing_interval :
  StrictMonoOn f (Set.Iio (-2)) ∧
  ∀ a b, a < b → b < -2 → ¬StrictMonoOn f (Set.Icc a b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l225_22551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_in_S_probability_is_one_l225_22542

-- Define the set S
def S : Set ℂ := {z | Complex.abs z.re ≤ 1 ∧ Complex.abs z.im ≤ 1}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + 1/2 * Complex.I) * z

-- Theorem statement
theorem transform_in_S (z : ℂ) : z ∈ S → transform z ∈ S := by
  sorry

-- Probability is 1 if the above theorem holds for all z in S
theorem probability_is_one : ∀ z ∈ S, transform z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_in_S_probability_is_one_l225_22542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_parrots_count_l225_22564

/-- The number of yellow parrots on Tropical Island -/
def yellow_parrots : ℕ := 30

/-- The total number of parrots on Tropical Island -/
def total_parrots : ℕ := 150

/-- The fraction of red parrots on Tropical Island -/
def red_fraction : ℚ := 4/5

theorem yellow_parrots_count :
  yellow_parrots = total_parrots - (red_fraction * ↑total_parrots).floor := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_parrots_count_l225_22564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selling_price_l225_22595

/-- Represents the fruit sales scenario --/
structure FruitSales where
  costPrice : ℚ
  initialSellingPrice : ℚ
  initialDailySales : ℚ
  priceReduction : ℚ
  salesIncrease : ℚ
  desiredProfit : ℚ

/-- Calculates the daily profit based on selling price and sales volume --/
noncomputable def dailyProfit (fs : FruitSales) (sellingPrice : ℚ) : ℚ :=
  let priceReduction := fs.initialSellingPrice - sellingPrice
  let salesVolume := fs.initialDailySales + (priceReduction / fs.priceReduction) * fs.salesIncrease
  (sellingPrice - fs.costPrice) * salesVolume

/-- Theorem stating that the calculated selling price achieves the desired profit --/
theorem optimal_selling_price (fs : FruitSales) 
    (h1 : fs.costPrice = 22)
    (h2 : fs.initialSellingPrice = 38)
    (h3 : fs.initialDailySales = 160)
    (h4 : fs.priceReduction = 3)
    (h5 : fs.salesIncrease = 120)
    (h6 : fs.desiredProfit = 3640) :
    ∃ (sellingPrice : ℚ), sellingPrice = 29 ∧ dailyProfit fs sellingPrice = fs.desiredProfit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selling_price_l225_22595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_count_theorem_l225_22594

theorem region_count_theorem : 
  ∀ h s : ℕ, 
    h > 0 → s > 0 →
    (s^2 + s)/2 + h*(s + 1) + 1 = 1992 ↔ 
    (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_count_theorem_l225_22594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_solutions_l225_22568

open Matrix

variable (a b c : ℝ)

def A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a - x, c, b],
    ![c, b - x, a],
    ![b, a, c - x]]

theorem det_A_solutions (x : ℝ) :
  det (A a b c x) = 0 ↔
  (x = a + b + c) ∨
  (x = Real.sqrt (a^2 + b^2 + c^2 - a*b - a*c - b*c)) ∨
  (x = -Real.sqrt (a^2 + b^2 + c^2 - a*b - a*c - b*c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_solutions_l225_22568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l225_22562

-- Define the curves
noncomputable def f (x : ℝ) : ℝ := x ^ 2003
noncomputable def g (x : ℝ) : ℝ := x ^ (1 / 2003)

-- Define the area function
noncomputable def area : ℝ := ∫ x in (0)..(1), g x - f x

-- Theorem statement
theorem area_between_curves : area = 1001 / 1002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l225_22562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l225_22552

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 + 3 * tan x + 6 / tan x + 4 / (tan x ^ 2) - 1

-- State the theorem
theorem f_minimum_value {x : ℝ} (hx : 0 < x ∧ x < π / 2) : 
  f x ≥ 3 + 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l225_22552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enrique_commission_l225_22563

/-- Calculates the total commission for Enrique based on his sales and commission structure --/
noncomputable def calculate_commission (mens_suits_price : ℚ) (mens_suits_quantity : ℕ)
                         (womens_blouses_price : ℚ) (womens_blouses_quantity : ℕ)
                         (mens_ties_price : ℚ) (mens_ties_quantity : ℕ)
                         (womens_dresses_price : ℚ) (womens_dresses_quantity : ℕ)
                         (base_commission_rate : ℚ) (higher_commission_rate : ℚ)
                         (womens_clothing_bonus_rate : ℚ) (sales_tax_rate : ℚ)
                         (commission_threshold : ℚ) : ℚ :=
  let total_sales := mens_suits_price * mens_suits_quantity +
                     womens_blouses_price * womens_blouses_quantity +
                     mens_ties_price * mens_ties_quantity +
                     womens_dresses_price * womens_dresses_quantity
  let womens_clothing_sales := womens_blouses_price * womens_blouses_quantity +
                               womens_dresses_price * womens_dresses_quantity
  let base_commission := min commission_threshold total_sales * base_commission_rate
  let higher_commission := max 0 (total_sales - commission_threshold) * higher_commission_rate
  let womens_bonus := womens_clothing_sales * womens_clothing_bonus_rate
  base_commission + higher_commission + womens_bonus

/-- Theorem stating that Enrique's commission for the given sales is $298.00 --/
theorem enrique_commission :
  calculate_commission 600 2 50 6 30 4 150 3 (1/10) (3/20) (1/20) (3/50) 1000 = 298 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enrique_commission_l225_22563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xz_intersection_radius_l225_22553

-- Define the sphere
noncomputable def sphere_center : ℝ × ℝ × ℝ := (3, 5, -8)
noncomputable def sphere_radius : ℝ := 2 * Real.sqrt 17

-- Define the intersection circles
def xy_intersection_center : ℝ × ℝ × ℝ := (3, 5, 0)
def xy_intersection_radius : ℝ := 2

def xz_intersection_center : ℝ × ℝ × ℝ := (0, 5, -8)

-- Theorem statement
theorem xz_intersection_radius :
  sphere_radius = 2 * Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xz_intersection_radius_l225_22553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_wall_leftover_space_l225_22521

/-- Calculates the leftover space on a library wall given the wall length and item dimensions -/
theorem library_wall_leftover_space
  (wall_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (chair_length : ℝ)
  (h_wall : wall_length = 30)
  (h_desk : desk_length = 2)
  (h_bookcase : bookcase_length = 1.5)
  (h_chair : chair_length = 0.5) :
  let set_length := desk_length + bookcase_length + chair_length
  let num_complete_sets := ⌊wall_length / set_length⌋
  let occupied_length := (↑num_complete_sets : ℝ) * set_length
  wall_length - occupied_length = 2 := by
  sorry

#check library_wall_leftover_space

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_wall_leftover_space_l225_22521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_payment_is_544_62_l225_22575

-- Define the payment schedule
def total_installments : ℕ := 104
def first_payment_amount : ℚ := 420
def second_payment_amount : ℚ := first_payment_amount + 75
def third_payment_amount : ℚ := second_payment_amount + 50
def fourth_payment_amount : ℚ := third_payment_amount + 100

def first_segment_count : ℕ := 16
def second_segment_count : ℕ := 24
def third_segment_count : ℕ := 32
def fourth_segment_count : ℕ := 32

-- Calculate total payments
def total_payments : ℚ := 
  first_payment_amount * first_segment_count +
  second_payment_amount * second_segment_count +
  third_payment_amount * third_segment_count +
  fourth_payment_amount * fourth_segment_count

-- Helper function to round to two decimal places
def round_to_two_decimals (q : ℚ) : ℚ :=
  (q * 100).floor / 100

-- Theorem statement
theorem average_payment_is_544_62 :
  round_to_two_decimals (total_payments / total_installments) = 544.62 := by
  sorry

#eval round_to_two_decimals (total_payments / total_installments)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_payment_is_544_62_l225_22575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_line_l225_22555

-- Define the polar equation
noncomputable def polar_equation (θ : ℝ) : ℝ := 1 / (2 * Real.sin θ - 3 * Real.cos θ)

-- Define the transformation from polar to Cartesian coordinates
noncomputable def to_cartesian (r θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

-- Theorem statement
theorem polar_equation_is_line :
  ∃ (a b c : ℝ), ∀ (θ : ℝ), 
    let (x, y) := to_cartesian (polar_equation θ) θ
    a * x + b * y = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_line_l225_22555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l225_22592

theorem no_solution_exists : ¬∃ (n k : ℕ+), (n ∣ k^(n:ℕ) - 1) ∧ Nat.gcd n.val (k.val - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l225_22592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l225_22531

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the propositions
def prop1 (Line Plane : Type) 
  (perpendicular : Line → Plane → Prop) 
  (parallel_lines : Line → Line → Prop) : Prop :=
  ∀ (a b : Line) (α : Plane),
    perpendicular a α → perpendicular b α → parallel_lines a b

def prop2 (Line Plane : Type) 
  (parallel_line_plane : Line → Plane → Prop) 
  (parallel_lines : Line → Line → Prop) : Prop :=
  ∀ (a b : Line) (α : Plane),
    parallel_line_plane a α → parallel_line_plane b α → parallel_lines a b

def prop3 (Line Plane : Type) 
  (perpendicular : Line → Plane → Prop) 
  (parallel_planes : Plane → Plane → Prop) : Prop :=
  ∀ (a : Line) (α β : Plane),
    perpendicular a α → perpendicular a β → parallel_planes α β

def prop4 (Line Plane : Type) 
  (parallel_line_plane : Line → Plane → Prop) 
  (parallel_planes : Plane → Plane → Prop) : Prop :=
  ∀ (b : Line) (α β : Plane),
    parallel_line_plane b α → parallel_line_plane b β → parallel_planes α β

-- The theorem to prove
theorem exactly_two_props_true :
  (prop1 Line Plane perpendicular parallel_lines ∧ 
   ¬prop2 Line Plane parallel_line_plane parallel_lines ∧ 
   prop3 Line Plane perpendicular parallel_planes ∧ 
   ¬prop4 Line Plane parallel_line_plane parallel_planes) ∨
  (prop1 Line Plane perpendicular parallel_lines ∧ 
   ¬prop2 Line Plane parallel_line_plane parallel_lines ∧ 
   ¬prop3 Line Plane perpendicular parallel_planes ∧ 
   prop4 Line Plane parallel_line_plane parallel_planes) ∨
  (prop1 Line Plane perpendicular parallel_lines ∧ 
   prop2 Line Plane parallel_line_plane parallel_lines ∧ 
   ¬prop3 Line Plane perpendicular parallel_planes ∧ 
   ¬prop4 Line Plane parallel_line_plane parallel_planes) ∨
  (¬prop1 Line Plane perpendicular parallel_lines ∧ 
   prop2 Line Plane parallel_line_plane parallel_lines ∧ 
   prop3 Line Plane perpendicular parallel_planes ∧ 
   ¬prop4 Line Plane parallel_line_plane parallel_planes) ∨
  (¬prop1 Line Plane perpendicular parallel_lines ∧ 
   prop2 Line Plane parallel_line_plane parallel_lines ∧ 
   ¬prop3 Line Plane perpendicular parallel_planes ∧ 
   prop4 Line Plane parallel_line_plane parallel_planes) ∨
  (¬prop1 Line Plane perpendicular parallel_lines ∧ 
   ¬prop2 Line Plane parallel_line_plane parallel_lines ∧ 
   prop3 Line Plane perpendicular parallel_planes ∧ 
   prop4 Line Plane parallel_line_plane parallel_planes) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l225_22531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ef_length_l225_22582

-- Define the rectangle ABCD
def rectangle_ABCD (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  ‖A - B‖ = 9 ∧ ‖B - C‖ = 12 ∧
  ‖A - B‖ = ‖C - D‖ ∧ ‖B - C‖ = ‖A - D‖ ∧
  ‖A - C‖^2 = ‖A - B‖^2 + ‖B - C‖^2

-- Define the triangle DEF
def triangle_DEF (D E F : EuclideanSpace ℝ (Fin 2)) : Prop :=
  D ≠ E ∧ E ≠ F ∧ F ≠ D

-- Define area functions (these would need to be properly implemented)
noncomputable def area_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry
noncomputable def area_rectangle (A B C D : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem ef_length (A B C D E F : EuclideanSpace ℝ (Fin 2)) :
  rectangle_ABCD A B C D →
  triangle_DEF D E F →
  ‖D - E‖ = ‖D - F‖ →
  area_triangle D E F = (1/3) * area_rectangle A B C D →
  ‖E - F‖ = 12 :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ef_length_l225_22582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l225_22585

def is_lattice_point (x y : ℤ) : Prop := True

def on_hyperbola (x y : ℤ) : Prop := x^2 - 2*y^2 = 2000^2

theorem lattice_points_on_hyperbola :
  ∃! (points : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ points ↔ is_lattice_point x y ∧ on_hyperbola x y) ∧
    points.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_hyperbola_l225_22585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_l225_22538

/-- Two lines in ℝ³ given by their parametric equations -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

/-- Two lines are perpendicular if their direction vectors are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  dot_product l1.direction l2.direction = 0

/-- The specific lines from the problem -/
def line1 (k : ℝ) : Line3D :=
  { point := fun i => [1, 2, 3].get i
    direction := fun i => [2, -1, k].get i }

def line2 (k : ℝ) : Line3D :=
  { point := fun i => [4, 5, 6].get i
    direction := fun i => [k, 3, 2].get i }

/-- The theorem to be proved -/
theorem lines_perpendicular_iff (k : ℝ) :
  perpendicular (line1 k) (line2 k) ↔ k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_iff_l225_22538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_fraction_after_cuts_l225_22587

-- Define a cube
structure Cube where
  volume : ℝ
  volume_positive : volume > 0

-- Define a plane that cuts the cube into two equal parts
def FirstCutPlane (c : Cube) : Set Cube := sorry

-- Define a plane that cuts one half of the cube into two parts with a 2:1 ratio
def SecondCutPlane (c : Cube) : Set Cube := sorry

-- Define the volume of a part of the cube
noncomputable def Volume (part : Set Cube) : ℝ := sorry

-- Define the piece containing the specific vertex after both cuts
def PieceWithVertex (c : Cube) : Set Cube := sorry

-- Theorem statement
theorem volume_fraction_after_cuts (c : Cube) :
  Volume (PieceWithVertex c) = (1 : ℝ) / 3 * c.volume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_fraction_after_cuts_l225_22587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l225_22577

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M : Set.compl M ∩ U = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l225_22577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l225_22523

-- Define the propositions p and q as functions of a
noncomputable def p (a : ℝ) : Prop := ∀ x, a * x^2 - 4*x + a > 0

def q (a : ℝ) : Prop := ∀ x, x < -1 → 2*x^2 + x > 2 + a*x

-- Define the main theorem
theorem range_of_a :
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∃ a : ℝ, 1 ≤ a ∧ a ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l225_22523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_repeat_l225_22544

/-- Represents a line of 2021 integers --/
def Line := Fin 2021 → ℕ

/-- Generates the next line based on the current line --/
def nextLine (l : Line) : Line :=
  fun i => (Finset.univ.filter (fun j => l j = l i)).card

/-- The sequence of lines generated by repeatedly applying nextLine --/
def lineSequence (initial : Line) : ℕ → Line
  | 0 => initial
  | n + 1 => nextLine (lineSequence initial n)

/-- The theorem stating that two consecutive identical lines will eventually occur --/
theorem eventual_repeat (initial : Line) :
  ∃ n : ℕ, lineSequence initial n = lineSequence initial (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_repeat_l225_22544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_element_l225_22541

def matrix : Matrix (Fin 6) (Fin 6) ℕ := ![
  ![12, 9, 5, 4, 7, 3],
  ![14, 8, 19, 15, 11, 10],
  ![10, 4, 6, 8, 12, 14],
  ![16, 5, 21, 18, 2, 1],
  ![9, 3, 7, 13, 5, 6],
  ![11, 2, 8, 10, 4, 9]
]

theorem no_max_min_element : ¬ ∃ (i j : Fin 6),
  (∀ k : Fin 6, matrix i j ≥ matrix k j) ∧
  (∀ l : Fin 6, matrix i j ≤ matrix i l) := by
  sorry

#check no_max_min_element

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_min_element_l225_22541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_strong_triples_coprime_sums_l225_22507

/-- A triple of positive integers is strong if for each integer m > 1,
    the sum of the triple does not divide the sum of their m-th powers. -/
def IsStrong (triple : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := triple
  ∀ m : ℕ, m > 1 → ¬((a + b + c) ∣ (a^m + b^m + c^m))

/-- The sum of a triple of natural numbers -/
def TripleSum (triple : ℕ × ℕ × ℕ) : ℕ :=
  let (a, b, c) := triple
  a + b + c

/-- There exists an infinite collection of strong triples with pairwise coprime sums -/
theorem infinite_strong_triples_coprime_sums :
  ∃ f : ℕ → ℕ × ℕ × ℕ,
    (∀ n, IsStrong (f n)) ∧
    (∀ n k, n ≠ k → Nat.Coprime (TripleSum (f n)) (TripleSum (f k))) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_strong_triples_coprime_sums_l225_22507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l225_22537

noncomputable def k (x : ℝ) : ℝ := 1 / (x + 9) + 1 / (x^2 + 9*x + 20) + 1 / (x^3 + 9)

theorem domain_of_k :
  {x : ℝ | ∃ y, k x = y} = 
    Set.univ \ {-9, -5, -4, -(9 : ℝ)^(1/3)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l225_22537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l225_22558

theorem sqrt_calculations :
  (Real.sqrt 18 / 3) * Real.sqrt 6 = 2 * Real.sqrt 3 ∧
  (Real.sqrt 18 + 3) / Real.sqrt 3 - 6 * Real.sqrt (3/2) = Real.sqrt 3 - 2 * Real.sqrt 6 ∧
  (Real.sqrt 7 + 2 * Real.sqrt 2) * (Real.sqrt 7 - 2 * Real.sqrt 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l225_22558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_sum_of_extrema_l225_22566

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_sum_of_extrema (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ max (f a 0) (f a 1) ∧ f a x ≥ min (f a 0) (f a 1)) →
  (f a 0 + f a 1 = 3) →
  a = 2 := by
  intro h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_sum_of_extrema_l225_22566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l225_22590

/-- The function f(x) = (x-2)/(x^2+4x-12) -/
noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x^2 + 4*x - 12)

/-- The number of vertical asymptotes of f -/
def num_vertical_asymptotes : ℕ := 1

/-- Theorem stating that f has exactly one vertical asymptote -/
theorem f_has_one_vertical_asymptote :
  ∃! (a : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (x : ℝ), 0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_vertical_asymptote_l225_22590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l225_22536

theorem largest_power_of_18_dividing_30_factorial :
  (∃ n : ℕ, 18^n ∣ (Nat.factorial 30) ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ (Nat.factorial 30))) ∧
  (∀ n : ℕ, (18^n ∣ (Nat.factorial 30) ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ (Nat.factorial 30))) → n = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l225_22536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l225_22598

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line on which P lies
def line_P (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the angle condition
def angle_condition (x y : ℝ) : Prop :=
  ∃ (qx qy : ℝ), circle_C qx qy ∧ 
  Real.arccos ((x * qx + y * qy) / Real.sqrt (x^2 + y^2)) = Real.pi / 6

-- Theorem statement
theorem x_range (x y : ℝ) :
  circle_C x y ∧ line_P x y ∧ angle_condition x y →
  0 ≤ x ∧ x ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l225_22598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_is_1955_l225_22579

/-- The marked price of a product given its cost price, selling price percentage, and profit margin. -/
noncomputable def marked_price (cost_price : ℝ) (selling_price_percentage : ℝ) (profit_margin : ℝ) : ℝ :=
  cost_price * (1 + profit_margin) / selling_price_percentage

/-- Theorem stating that the marked price is 1955 yuan given the specified conditions. -/
theorem marked_price_is_1955 :
  marked_price 1360 0.8 0.15 = 1955 := by
  -- Unfold the definition of marked_price
  unfold marked_price
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_is_1955_l225_22579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_birth_rate_l225_22574

/-- Represents the average birth rate in a city -/
def average_birth_rate : ℝ := sorry

/-- The death rate in the city (people per two seconds) -/
def death_rate : ℝ := 2

/-- The net population increase in one day -/
def net_increase : ℕ := 216000

/-- The number of two-second intervals in a day -/
def intervals_per_day : ℕ := 43200

/-- Theorem stating the average birth rate in the city -/
theorem city_birth_rate :
  (average_birth_rate - death_rate) * (intervals_per_day : ℝ) = net_increase →
  average_birth_rate = 7 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_birth_rate_l225_22574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_middle_y_coord_l225_22539

/-- Given three points in a 2D plane, proves that if they are collinear, 
    then the y-coordinate of the middle point is 2. -/
theorem collinear_points_middle_y_coord 
  (M N P : ℝ × ℝ) 
  (hM : M = (2, -1)) 
  (hN : N = (4, 5)) 
  (hP : P.1 = 3) 
  (h_collinear : (N.2 - M.2) * (P.1 - M.1) = (P.2 - M.2) * (N.1 - M.1)) : 
  P.2 = 2 := by
  sorry

#check collinear_points_middle_y_coord

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_middle_y_coord_l225_22539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_f_eq_7_l225_22527

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 3 * x - 5

-- State the theorem
theorem unique_solution_f_f_eq_7 :
  ∃! x : ℝ, f (f x) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_f_eq_7_l225_22527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_group_frequency_l225_22503

theorem eighth_group_frequency 
  (total_size : ℕ)
  (num_groups : ℕ)
  (freq_1 freq_2 freq_3 freq_4 : ℕ)
  (freq_5_to_7 : ℚ)
  (h1 : total_size = 64)
  (h2 : num_groups = 8)
  (h3 : freq_1 = 5)
  (h4 : freq_2 = 7)
  (h5 : freq_3 = 11)
  (h6 : freq_4 = 13)
  (h7 : freq_5_to_7 = 1/8)
  : ∃ (freq_8 : ℕ), freq_8 = 4 ∧ 
    freq_1 + freq_2 + freq_3 + freq_4 + 3 * (freq_5_to_7 * total_size).floor + freq_8 = total_size := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_group_frequency_l225_22503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_machine_payoff_l225_22550

/-- Calculates the number of days until a coffee machine pays for itself. -/
def coffee_machine_payoff_days (
  machine_cost : ℚ)
  (discount : ℚ)
  (daily_home_cost : ℚ)
  (daily_coffees : ℕ)
  (coffee_price : ℚ) : ℕ :=
  let actual_cost := machine_cost - discount
  let daily_savings := daily_coffees * coffee_price - daily_home_cost
  (actual_cost / daily_savings).ceil.toNat

/-- Theorem stating that the coffee machine pays for itself in 36 days. -/
theorem coffee_machine_payoff :
  coffee_machine_payoff_days 200 20 3 2 4 = 36 := by
  sorry

#eval coffee_machine_payoff_days 200 20 3 2 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_machine_payoff_l225_22550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_12_not_sqrt_2_multiple_l225_22515

noncomputable section

-- Define the square roots we're considering
def sqrt_half : ℝ := Real.sqrt (1/2)
def sqrt_8 : ℝ := Real.sqrt 8
def sqrt_12 : ℝ := Real.sqrt 12
def sqrt_18 : ℝ := Real.sqrt 18

-- Define a predicate for whether a real number can be expressed as a * √2 for some rational a
def is_sqrt_2_multiple (x : ℝ) : Prop :=
  ∃ (a : ℚ), x = a * Real.sqrt 2

-- Theorem stating that √12 is the only one that cannot be expressed as a * √2
theorem sqrt_12_not_sqrt_2_multiple :
  is_sqrt_2_multiple sqrt_half ∧
  is_sqrt_2_multiple sqrt_8 ∧
  ¬is_sqrt_2_multiple sqrt_12 ∧
  is_sqrt_2_multiple sqrt_18 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_12_not_sqrt_2_multiple_l225_22515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_apothem_angle_l225_22573

/-- Regular triangular pyramid -/
structure RegularTriangularPyramid where
  /-- Tangent of the angle between a lateral face and the base plane -/
  k : ℝ

/-- Angle between lateral edge and apothem of the opposite face -/
noncomputable def angle_between_lateral_edge_and_apothem (pyramid : RegularTriangularPyramid) : ℝ :=
  Real.arctan (3 * pyramid.k / (pyramid.k^2 - 2))

/-- 
Given a regular triangular pyramid where the tangent of the angle between 
a lateral face and the base plane is k, the tangent of the angle between 
a lateral edge and the apothem of the opposite face is 3k / (k^2 - 2).
-/
theorem lateral_edge_apothem_angle (pyramid : RegularTriangularPyramid) :
  Real.tan (angle_between_lateral_edge_and_apothem pyramid) = 3 * pyramid.k / (pyramid.k^2 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_apothem_angle_l225_22573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_min_distance_l225_22524

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : Real) : Real := 4 * Real.cos θ

-- Define the line l in parametric form
noncomputable def line_l (t : Real) : Real × Real := (-3 + Real.sqrt 3 / 2 * t, 1 / 2 * t)

-- State the theorem
theorem projection_and_min_distance :
  -- Part 1: Projection point P
  ∃ (r θ : Real), 
    -- P is on line l
    (∃ t, line_l t = (r * Real.cos θ, r * Real.sin θ)) ∧ 
    -- P is perpendicular to line l
    (∃ t, (3 * (-3 + Real.sqrt 3 / 2 * t) + Real.sqrt 3 * (1 / 2 * t) = 0)) ∧
    -- P has the specified polar coordinates
    r = 3 / 2 ∧ θ = 2 * Real.pi / 3 ∧
  -- Part 2: Minimum distance
  (∀ θ t, 
    let (x, y) := line_l t
    let ρ := curve_C θ
    (ρ * Real.cos θ - x) ^ 2 + (ρ * Real.sin θ - y) ^ 2 ≥ (1 / 2) ^ 2) ∧
  (∃ θ t, 
    let (x, y) := line_l t
    let ρ := curve_C θ
    (ρ * Real.cos θ - x) ^ 2 + (ρ * Real.sin θ - y) ^ 2 = (1 / 2) ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_min_distance_l225_22524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_medians_angle_bound_l225_22580

/-- A median of a triangle -/
def median (triangle : Set (ℝ × ℝ)) (vertex : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The angle at a specific vertex of a triangle -/
noncomputable def angle_of (triangle : Set (ℝ × ℝ)) (vertex : ℝ × ℝ) : ℝ := sorry

/-- Perpendicularity of two line segments -/
def perpendicular (s1 s2 : Set (ℝ × ℝ)) : Prop := sorry

/-- Given a triangle ABC with medians AD and BE that are perpendicular,
    the angle C is less than or equal to arccos(4/5) -/
theorem perpendicular_medians_angle_bound (A B C : ℝ × ℝ) :
  let triangle := {A, B, C}
  let AD := median triangle A
  let BE := median triangle B
  perpendicular AD BE →
  angle_of triangle C ≤ Real.arccos (4/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_medians_angle_bound_l225_22580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_largest_smallest_sum_l225_22559

/-- A triangle with interior angles forming an arithmetic sequence -/
structure ArithmeticTriangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  angle_sum : A + B + C = 180
  arithmetic_seq : 2 * B = A + C

/-- The sum of the largest and smallest angles in an arithmetic triangle is 120° -/
theorem arithmetic_triangle_largest_smallest_sum (t : ArithmeticTriangle) : 
  max t.A (max t.B t.C) + min t.A (min t.B t.C) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_largest_smallest_sum_l225_22559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_average_speed_l225_22509

/-- Calculates the average speed given total time, rest time, and total distance. -/
noncomputable def average_speed (total_time : ℝ) (rest_time : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance / (total_time - rest_time)

/-- Theorem stating that for Sam's journey, the average speed is 200/6 miles per hour. -/
theorem sam_average_speed :
  let total_time : ℝ := 7
  let rest_time : ℝ := 1
  let total_distance : ℝ := 200
  average_speed total_time rest_time total_distance = 200 / 6 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

#eval (200 : ℚ) / 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_average_speed_l225_22509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l225_22535

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.A < Real.pi ∧
  t.B > 0 ∧ t.B < Real.pi ∧
  t.C > 0 ∧ t.C < Real.pi ∧
  3 * (t.b^2 + t.c^2) = 3 * t.a^2 + 2 * t.b * t.c ∧
  t.a = 3/2 ∧
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 2 / 2 ∧
  t.b > t.c

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = 2 * Real.sqrt 2 / 3 ∧ t.b = 3/2 ∧ t.c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l225_22535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_destruction_shots_l225_22543

/-- Represents a checkerboard field -/
structure CheckerboardField where
  size : ℕ
  cells : ℕ

/-- Represents a tank on the field -/
structure Tank where
  position : ℕ × ℕ

/-- Represents a shot fired at the field -/
def Shot := ℕ × ℕ

/-- The minimum number of shots required to guarantee destroying the tank -/
def min_shots_to_destroy (f : CheckerboardField) (t : Tank) : ℕ := 2521

/-- Theorem stating the minimum number of shots required to destroy the tank -/
theorem tank_destruction_shots (f : CheckerboardField) (t : Tank) :
  f.size = 41 ∧ f.cells = 41 * 41 →
  min_shots_to_destroy f t = 2521 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_destruction_shots_l225_22543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l225_22588

/-- A parabola with equation y² = 12x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun x y => y^2 = 12 * x

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (3, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_to_focus (p : Parabola) (a : PointOnParabola p) (h : a.x = 4) :
  distance (a.x, a.y) (focus p) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l225_22588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_l225_22548

/-- The cost to park a car in a certain parking garage for up to 2 hours -/
def C : ℝ := sorry

/-- The cost for each hour in excess of 2 hours -/
def excess_cost : ℝ := 1.75

/-- The average cost per hour for 9 hours of parking -/
def average_cost : ℝ := 2.4722222222222223

/-- The total parking duration in hours -/
def total_hours : ℕ := 9

/-- Theorem stating that the cost for up to 2 hours of parking is $10 -/
theorem parking_cost : C = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_l225_22548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l225_22569

theorem sin_plus_cos_for_point (α : ℝ) :
  let x : ℝ := -5
  let y : ℝ := 12
  let r : ℝ := Real.sqrt (x^2 + y^2)
  (Real.sin α = y / r) → (Real.cos α = x / r) → Real.sin α + Real.cos α = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l225_22569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cisSum_angle_l225_22534

-- Define the type for complex numbers in polar form
def PolarComplex := ℝ × ℝ

-- Define the cis function
noncomputable def cis (θ : ℝ) : PolarComplex := (Real.cos θ, Real.sin θ)

-- Define the sum of cis from 65° to 157° with 8° steps
noncomputable def cisSum : PolarComplex :=
  let angles : List ℝ := List.range 12 |>.map (fun n => (65 + 8 * n) * Real.pi / 180)
  angles.foldl (fun acc θ => (acc.1 + (cis θ).1, acc.2 + (cis θ).2)) (0, 0)

-- Define the theorem
theorem cisSum_angle : 
  ∃ (r : ℝ) (θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
  cisSum = (r * Real.cos (111 * Real.pi / 180), r * Real.sin (111 * Real.pi / 180)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cisSum_angle_l225_22534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_range_condition_l225_22545

/-- Sequence a_n defined recursively --/
noncomputable def a (c : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => c - (1 / a c n)

/-- Sequence b_n defined in terms of a_n --/
noncomputable def b (n : ℕ) : ℝ := 1 / (a (5/2) n - 2)

/-- Theorem for sequence equality --/
theorem sequence_equality :
  ∀ n : ℕ, b n = -(1/3) * (4^(n-1) + 2) := by
  sorry

/-- Theorem for range condition --/
theorem range_condition :
  ∀ c : ℝ, (∀ n : ℕ, a c n < a c (n+1) ∧ a c (n+1) < 3) ↔ (2 < c ∧ c ≤ 10/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_range_condition_l225_22545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l225_22586

theorem equation_solution : ∃ x : ℝ, (2:ℝ)^(3*x) * (8:ℝ)^(2*x) = (512:ℝ)^3 ∧ x = 3 := by
  use 3
  apply And.intro
  · -- Prove the equation
    simp [Real.rpow_mul, Real.rpow_add]
    norm_num
  · -- Prove x = 3
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l225_22586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_member_with_eight_nights_l225_22526

/-- A board game club with members and game nights. -/
structure GameClub where
  members : Finset Nat
  game_nights : Finset Nat
  attended : Nat → Nat → Bool

/-- The properties of the game club. -/
def ValidGameClub (club : GameClub) : Prop :=
  (club.members.card = 50) ∧
  (∀ m₁ m₂, m₁ ∈ club.members → m₂ ∈ club.members → m₁ ≠ m₂ → 
    ∃! n, n ∈ club.game_nights ∧ club.attended m₁ n ∧ club.attended m₂ n) ∧
  (∀ n, n ∈ club.game_nights → ∃ m, m ∈ club.members ∧ ¬club.attended m n)

/-- The theorem to be proved. -/
theorem exists_member_with_eight_nights (club : GameClub) (h : ValidGameClub club) :
  ∃ m, m ∈ club.members ∧ (club.game_nights.filter (fun n => club.attended m n)).card ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_member_with_eight_nights_l225_22526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2018_equals_neg_one_l225_22501

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / mySequence n

theorem mySequence_2018_equals_neg_one : mySequence 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_2018_equals_neg_one_l225_22501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_tripling_l225_22505

theorem marble_fraction_after_tripling (n : ℕ) (hn : n > 0) : 
  let initial_blue : ℚ := 4 / 7
  let initial_green : ℚ := 1 - initial_blue
  let new_green : ℚ := 3 * initial_green * n
  let new_total : ℚ := initial_blue * n + new_green
  new_green / new_total = 9 / 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_tripling_l225_22505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_perimeter_lower_bound_l225_22533

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : 0 < width
  height_pos : 0 < height

/-- A quadrilateral inscribed in a rectangle -/
structure InscribedQuadrilateral (rect : Rectangle) where
  vertices : Fin 4 → ℝ × ℝ
  on_sides : ∀ i, (vertices i).1 = 0 ∨ (vertices i).1 = rect.width ∨
                  (vertices i).2 = 0 ∨ (vertices i).2 = rect.height
  distinct_sides : ∀ i j, i ≠ j → 
    (((vertices i).1 = 0 ∧ (vertices j).1 = rect.width) ∨
     ((vertices i).1 = rect.width ∧ (vertices j).1 = 0) ∨
     ((vertices i).2 = 0 ∧ (vertices j).2 = rect.height) ∨
     ((vertices i).2 = rect.height ∧ (vertices j).2 = 0))

/-- The perimeter of a quadrilateral -/
noncomputable def perimeter (rect : Rectangle) (quad : InscribedQuadrilateral rect) : ℝ :=
  Finset.sum (Finset.range 4) (fun i => 
    Real.sqrt (((quad.vertices (i + 1)).1 - (quad.vertices i).1) ^ 2 +
               ((quad.vertices (i + 1)).2 - (quad.vertices i).2) ^ 2))

/-- The diagonal of a rectangle -/
noncomputable def diagonal (rect : Rectangle) : ℝ :=
  Real.sqrt (rect.width ^ 2 + rect.height ^ 2)

theorem inscribed_quadrilateral_perimeter_lower_bound 
  (rect : Rectangle) (quad : InscribedQuadrilateral rect) :
  perimeter rect quad ≥ 2 * diagonal rect := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_perimeter_lower_bound_l225_22533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_to_pipes_l225_22572

noncomputable def shortest_distance : ℝ := 4 * Real.sqrt 5

def start_point : ℝ × ℝ := (5, 1)

def line1 (x : ℝ) : ℝ := x

def line2 : ℝ := 7

noncomputable def path_length (path : List (ℝ × ℝ)) : ℝ :=
  path.zip path.tail
    |>.map (λ ((x₁, y₁), (x₂, y₂)) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2))
    |>.sum

theorem shortest_path_to_pipes :
  ∃ (path : List (ℝ × ℝ)),
    path.head? = some start_point ∧
    path.getLast? = some start_point ∧
    (∃ (p : ℝ × ℝ), p ∈ path ∧ p.1 = p.2) ∧
    (∃ (p : ℝ × ℝ), p ∈ path ∧ p.1 = line2) ∧
    (path.length > 1) ∧
    (∀ (other_path : List (ℝ × ℝ)),
      other_path.head? = some start_point →
      other_path.getLast? = some start_point →
      (∃ (p : ℝ × ℝ), p ∈ other_path ∧ p.1 = p.2) →
      (∃ (p : ℝ × ℝ), p ∈ other_path ∧ p.1 = line2) →
      (other_path.length > 1) →
      path_length path ≤ path_length other_path) ∧
    path_length path = shortest_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_to_pipes_l225_22572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_count_l225_22549

theorem ordered_pairs_count (n : ℕ) (h : n = 2541) 
  (factorization : n = 3^1 * 13^2 * 5^1) : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ 0 < p.1 ∧ 0 < p.2) (Finset.range (n+1) ×ˢ Finset.range (n+1))).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_count_l225_22549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l225_22528

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees -/
  vertex_angle : ℝ
  /-- The sphere is tangent to the sides of the cone and sits on the table -/
  sphere_tangent : Bool

/-- The volume of a sphere -/
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

/-- Theorem stating the volume of the inscribed sphere -/
theorem inscribed_sphere_volume (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90)
  (h3 : cone.sphere_tangent = true) :
  ∃ (v : ℝ), v = sphere_volume (cone.base_diameter / 8) ∧ v = 288 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l225_22528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l225_22522

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem parallel_vectors_magnitude (y : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, y)
  parallel a b → magnitude (3 • a + b) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l225_22522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l225_22560

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the proposition P
def P (a : ℝ) : Prop := ∃! x, x ∈ Set.Icc 0 1 ∧ f a x = 0

-- Define the function g (y = a^x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the proposition Q
def Q (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ StrictMono (g a)

-- State the theorem
theorem main_theorem : ∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → 1 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l225_22560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_sum_l225_22554

/-- The parabola with equation x² = 8y -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 8 * p.2}

/-- The focus of the parabola x² = 8y -/
def F : ℝ × ℝ := (0, 2)

/-- Definition of a vector from the focus to a point -/
def FA (A : ℝ × ℝ) : ℝ × ℝ := (A.1 - F.1, A.2 - F.2)

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parabola_focus_sum (A B C : ℝ × ℝ) :
  A ∈ Parabola → B ∈ Parabola → C ∈ Parabola →
  FA A + FA B + FA C = (0, 0) →
  magnitude (FA A) + magnitude (FA B) + magnitude (FA C) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_sum_l225_22554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_number_assignment_l225_22578

-- Define the set of numbers
def NumberSet : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 11}

-- Define a type for the vertices of the tetrahedron
inductive Vertex : Type
| P : Vertex
| Q : Vertex
| R : Vertex
| S : Vertex

-- Define a type for the edges of the tetrahedron
inductive Edge : Type
| PQ : Edge
| PR : Edge
| PS : Edge
| QR : Edge
| QS : Edge
| RS : Edge

-- Function to assign numbers to vertices
def vertex_assignment : Vertex → ℕ := sorry

-- Function to assign numbers to edges
def edge_assignment : Edge → ℕ := sorry

-- Function to get the vertices of an edge
def Edge.vertices : Edge → Vertex × Vertex
| Edge.PQ => (Vertex.P, Vertex.Q)
| Edge.PR => (Vertex.P, Vertex.R)
| Edge.PS => (Vertex.P, Vertex.S)
| Edge.QR => (Vertex.Q, Vertex.R)
| Edge.QS => (Vertex.Q, Vertex.S)
| Edge.RS => (Vertex.R, Vertex.S)

-- Theorem statement
theorem tetrahedron_number_assignment
  (h1 : ∀ v : Vertex, vertex_assignment v ∈ NumberSet)
  (h2 : ∀ e : Edge, edge_assignment e ∈ NumberSet)
  (h3 : ∀ n : ℕ, n ∈ NumberSet → (∃! x : Vertex ⊕ Edge, (x.elim vertex_assignment edge_assignment) = n))
  (h4 : ∀ e : Edge, edge_assignment e = vertex_assignment (Edge.vertices e).1 + vertex_assignment (Edge.vertices e).2)
  (h5 : edge_assignment Edge.PQ = 9) :
  edge_assignment Edge.RS = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_number_assignment_l225_22578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circles_perimeter_is_2pi_l225_22517

/-- The perimeter of four quarter circles drawn outward at the vertices of a unit square -/
noncomputable def quarter_circles_perimeter : ℝ := 2 * Real.pi

/-- Theorem: The perimeter of four quarter circles drawn outward at the vertices of a unit square is 2π -/
theorem quarter_circles_perimeter_is_2pi :
  quarter_circles_perimeter = 2 * Real.pi := by
  -- Unfold the definition of quarter_circles_perimeter
  unfold quarter_circles_perimeter
  -- The equality now follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circles_perimeter_is_2pi_l225_22517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l225_22512

-- Define the points and circle
def A : ℝ × ℝ := (-2, 2)
def B : ℝ × ℝ := (0, A.2)  -- B is on y-axis with same y-coordinate as A
def C : ℝ × ℝ := (2, A.2)  -- C has same y-coordinate as A, and x-coordinate is positive 2
def D : ℝ × ℝ := (2, 6)
def E : ℝ × ℝ := (2, 2)
def O : ℝ × ℝ := (0, 0)  -- Assuming circle O is centered at origin

-- Define the radius of circle O
def radius_O : ℝ := 2

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the theorem
theorem tangent_circle_radius :
  ∃ (r : ℝ), 
    -- Triangle ABC is equilateral
    (distance A B = distance B C) ∧ (distance B C = distance C A) ∧
    -- B is on y-axis
    (B.1 = 0) ∧
    -- CB is parallel to x-axis
    (C.2 = B.2) ∧
    -- Circle with radius r is externally tangent to circle O
    (distance (A.1 - r, A.2 + r * Real.sqrt 3) O = radius_O + r) ∧
    -- Circle with radius r is tangent to AB
    (distance (A.1 - r, A.2 + r * Real.sqrt 3) A = r) ∧
    -- Circle with radius r is tangent to DE
    (distance (A.1 - r, A.2 + r * Real.sqrt 3) D = r) ∧
    -- The radius is approximately 1.4726
    (abs (r - 1.4726) < 0.0001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_radius_l225_22512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l225_22514

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin (x + Real.pi/8)^2 + 2 * Real.sin (x + Real.pi/8) * Real.cos (x + Real.pi/8)

theorem f_properties :
  let period : ℝ := Real.pi
  let mono_increasing (k : ℤ) := Set.Icc (-Real.pi/2 + k * Real.pi) (k * Real.pi)
  let interval : Set ℝ := Set.Icc (-Real.pi/4) (3*Real.pi/8)
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ mono_increasing k, ∀ y ∈ mono_increasing k, x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ interval, f x ≤ Real.sqrt 2) ∧
  (∀ x ∈ interval, f x ≥ -1) ∧
  (∃ x ∈ interval, f x = Real.sqrt 2) ∧
  (∃ x ∈ interval, f x = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l225_22514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_chord_length_l225_22518

/-- The length of the chord intercepted by a circle on a line -/
theorem chord_length (r : ℝ) (a b c : ℝ) : 
  let circle := λ (x y : ℝ) ↦ x^2 + y^2 = r^2
  let line := λ (x y : ℝ) ↦ a*x + b*y + c = 0
  let d := |c| / Real.sqrt (a^2 + b^2)
  r > 0 → a^2 + b^2 ≠ 0 → 
  (∃ x y : ℝ, circle x y ∧ line x y) →
  ∃ l : ℝ, l^2 = 4 * (r^2 - d^2) ∧ 
    l = Real.sqrt (4 * r^2 - ((a^2 + b^2) * c^2) / (a^2 + b^2)) :=
by sorry

/-- The specific chord length for the given problem -/
theorem specific_chord_length : 
  ∃ l : ℝ, l = Real.sqrt 14 ∧
  l^2 = 4 * (4 - ((1^2 + 1^2) * (-1)^2) / (1^2 + 1^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_chord_length_l225_22518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_number_is_121_l225_22591

def customSequence : List Nat := [12, 13, 15, 17, 111, 113, 117, 119, 0, 129, 131]

def uses_only_digits_one_and_two (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 2

def follows_pattern (s : List Nat) : Prop :=
  ∀ i, i < s.length → i < s.length → uses_only_digits_one_and_two (s[i]!)

theorem ninth_number_is_121 (s : List Nat) (h1 : s = customSequence) (h2 : follows_pattern s) :
  s[8]! = 121 := by
  sorry

#eval customSequence[8]!

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_number_is_121_l225_22591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l225_22547

def a : Fin 2 → ℝ := ![(-1), 1]
def b : Fin 2 → ℝ := ![3, 1]

theorem projection_magnitude : 
  ‖(((a • b) / (a • a)) • a)‖ = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_magnitude_l225_22547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l225_22561

theorem inequality_proof (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (2 : ℝ)^(Real.sin x) + (2 : ℝ)^(Real.tan x) ≥ (2 : ℝ)^(x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l225_22561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l225_22532

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := sin x * cos x * (sin x + cos x)

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range (f ∘ (fun x => x * (π/4) - π/4)),
    -sqrt 3 / 9 ≤ y ∧ y ≤ sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l225_22532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l225_22599

/-- The length of a train given its crossing times over two structures -/
theorem train_length (tunnel_length platform_length tunnel_time platform_time train_length : ℝ) 
  (h1 : tunnel_length = 1200)
  (h2 : platform_length = 180)
  (h3 : tunnel_time = 45)
  (h4 : platform_time = 15)
  (h5 : (train_length + tunnel_length) / tunnel_time = (train_length + platform_length) / platform_time) :
  train_length = 330 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l225_22599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l225_22510

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = min) ∧
    max = 16 ∧ min = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l225_22510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_75_l225_22556

theorem distinct_prime_factors_of_75 : (Nat.factors 75).eraseDups.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_factors_of_75_l225_22556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_cyclist_catchup_l225_22513

/-- The time (in minutes) for a hiker to catch up with a cyclist, given their speeds and the cyclist's wait time --/
noncomputable def catchUpTime (hikerSpeed : ℝ) (cyclistSpeed : ℝ) (cyclistWaitTime : ℝ) : ℝ :=
  let cyclistDistance := cyclistSpeed * cyclistWaitTime / 60
  cyclistDistance / (hikerSpeed / 60)

theorem hiker_cyclist_catchup :
  let hikerSpeed : ℝ := 4  -- miles per hour
  let cyclistSpeed : ℝ := 10  -- miles per hour
  let cyclistWaitTime : ℝ := 5  -- minutes
  catchUpTime hikerSpeed cyclistSpeed cyclistWaitTime = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_cyclist_catchup_l225_22513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_area_ratio_l225_22502

/-- The ratio of the area of a rectangle to the area of a circle -/
noncomputable def area_ratio (rectangle_area : ℝ) (circle_radius : ℝ) : ℝ :=
  rectangle_area / (Real.pi * circle_radius^2)

/-- Theorem stating the ratio of areas for the given rectangle and circle -/
theorem rectangle_circle_area_ratio :
  let rectangle_area : ℝ := 50
  let circle_radius : ℝ := 5
  abs (area_ratio rectangle_area circle_radius - 0.637) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_area_ratio_l225_22502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l225_22567

-- Define the given parameters
noncomputable def interest_rate : ℝ := 0.15
noncomputable def time_period : ℝ := 2 + 4 / 12
noncomputable def compounding_frequency : ℝ := 1
noncomputable def compound_interest : ℝ := 2331.75

-- Define the compound interest formula
noncomputable def compound_interest_formula (P : ℝ) : ℝ :=
  P * ((1 + interest_rate / compounding_frequency) ^ (compounding_frequency * time_period) - 1)

-- State the theorem
theorem initial_amount_proof (P : ℝ) :
  compound_interest_formula P = compound_interest →
  ∃ ε > 0, |P - 5757.47| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l225_22567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l225_22529

theorem trigonometric_identities 
  (α β : Real)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : π/2 < β ∧ β < π)
  (h3 : Real.cos (α + π/4) = 1/3)
  (h4 : Real.cos (π/4 - β/2) = Real.sqrt 3/3) :
  Real.cos β = -4 * Real.sqrt 2/9 ∧ Real.cos (2*α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l225_22529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_phi_for_symmetry_l225_22511

/-- The original function f(x) = sin(2x + π/4) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

/-- The translated function g(x) = f(x - φ) -/
noncomputable def g (x φ : ℝ) : ℝ := f (x - φ)

/-- Theorem stating the minimum positive value of φ for y-axis symmetry -/
theorem min_positive_phi_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ x, g x φ = g (-x) φ) ∧
  (∀ ψ, ψ > 0 ∧ (∀ x, g x ψ = g (-x) ψ) → φ ≤ ψ) ∧
  φ = 3 * Real.pi / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_phi_for_symmetry_l225_22511
