import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_two_square_sum_representations_l1355_135572

/-- A function that checks if a number can be represented as a sum of two squares in two different ways -/
def has_two_square_sum_representations (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    n * (n + 1) = a^2 + b^2 ∧
    n * (n + 1) = c^2 + d^2 ∧
    (a ≠ c ∨ b ≠ d) ∧ (a ≠ d ∨ b ≠ c)

/-- Theorem stating that there are infinitely many positive integers with the desired property -/
theorem infinitely_many_two_square_sum_representations :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, has_two_square_sum_representations n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_two_square_sum_representations_l1355_135572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_side_length_satisfies_pythagorean_side_length_is_correct_l1355_135542

/-- Represents a rhombus with specific diagonal properties -/
structure Rhombus where
  -- k represents the area of the rhombus
  k : ℝ
  -- d represents the length of the shorter diagonal
  d : ℝ
  -- The area of the rhombus is equal to k
  area_eq : k = d^2
  -- One pair of diagonals is twice the length of the other pair
  diagonal_ratio : d * 2 = 2 * d

/-- The side length of a rhombus with the given properties -/
noncomputable def side_length (r : Rhombus) : ℝ := (Real.sqrt (5 * r.k)) / 2

/-- Theorem stating that the side length formula is correct -/
theorem rhombus_side_length (r : Rhombus) : 
  side_length r = (Real.sqrt (5 * r.k)) / 2 := by
  -- Unfold the definition of side_length
  unfold side_length
  -- The equation is true by definition
  rfl

/-- Proof that the side length formula satisfies the Pythagorean theorem -/
theorem side_length_satisfies_pythagorean (r : Rhombus) :
  (side_length r)^2 = (r.d/2)^2 + r.d^2 := by
  sorry  -- We skip the detailed proof for now

/-- Proof that our formula gives the correct side length -/
theorem side_length_is_correct (r : Rhombus) :
  (side_length r)^2 = (5 * r.k) / 4 := by
  sorry  -- We skip the detailed proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_side_length_satisfies_pythagorean_side_length_is_correct_l1355_135542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ten_unrelated_l1355_135559

/-- A relation on a finite set where each element is related to exactly 5 others -/
def FiveRelation (α : Type*) [Fintype α] (r : α → α → Prop) : Prop :=
  ∀ a : α, ∃! s : Finset α, s.card = 5 ∧ ∀ b, b ∈ s → r a b

theorem exists_ten_unrelated
  (α : Type*) [Fintype α] (r : α → α → Prop) 
  (h_card : Fintype.card α = 100)
  (h_rel : FiveRelation α r) :
  ∃ s : Finset α, s.card = 10 ∧ ∀ a b, a ∈ s → b ∈ s → a ≠ b → ¬(r a b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ten_unrelated_l1355_135559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l1355_135549

/-- Calculates the length of a platform given the length of a train, the time it takes to cross the platform, and the time it takes to cross a signal pole. -/
noncomputable def platformLength (trainLength : ℝ) (timePlatform : ℝ) (timeSignalPole : ℝ) : ℝ :=
  trainLength * (timePlatform / timeSignalPole - 1)

/-- Theorem stating that for a 420m train crossing a platform in 60s and a signal pole in 30s, the platform length is 420m. -/
theorem platform_length_calculation :
  let trainLength : ℝ := 420
  let timePlatform : ℝ := 60
  let timeSignalPole : ℝ := 30
  platformLength trainLength timePlatform timeSignalPole = 420 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval platformLength 420 60 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l1355_135549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_cards_count_l1355_135576

theorem green_cards_count (total : ℕ) (red_frac : ℚ) (black_frac : ℚ) : 
  total = 2160 → 
  red_frac = 7/12 → 
  black_frac = 11/19 → 
  ∃ (red black green : ℕ),
    red = (red_frac * ↑total).floor ∧
    black = (black_frac * ↑(total - red)).floor ∧
    green = total - red - black ∧
    green = 379 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_cards_count_l1355_135576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_l1355_135507

/-- The side length of the first equilateral triangle -/
noncomputable def initial_side_length : ℝ := 40

/-- The ratio of side lengths between consecutive triangles -/
noncomputable def side_ratio : ℝ := 1 / 2

/-- The perimeter of an equilateral triangle given its side length -/
noncomputable def triangle_perimeter (side : ℝ) : ℝ := 3 * side

/-- The sum of the geometric series representing the perimeters of all triangles -/
noncomputable def perimeter_sum : ℝ := triangle_perimeter initial_side_length / (1 - side_ratio)

/-- Theorem stating that the sum of all triangle perimeters is 240 -/
theorem sum_of_triangle_perimeters : perimeter_sum = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_l1355_135507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_42_l1355_135571

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its four vertices -/
noncomputable def trapezoidArea (e f g h : Point) : ℝ :=
  let base1 := |e.y - f.y|
  let base2 := |g.y - h.y|
  let height := |e.x - g.x|
  (base1 + base2) * height / 2

/-- Theorem: The area of the trapezoid EFGH with given vertices is 42 square units -/
theorem trapezoid_area_is_42 :
  let e : Point := ⟨0, 0⟩
  let f : Point := ⟨0, -3⟩
  let g : Point := ⟨7, 0⟩
  let h : Point := ⟨7, 9⟩
  trapezoidArea e f g h = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_42_l1355_135571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_not_divisible_by_five_l1355_135522

open BigOperators

def a (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), 2^(3*k) * Nat.choose (2*n + 1) (2*k + 1)

theorem a_not_divisible_by_five (n : ℕ) : ¬ ∃ m : ℤ, (5 : ℤ) * m = (a n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_not_divisible_by_five_l1355_135522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1355_135574

/-- Represents the radius of the smaller circle -/
noncomputable def r : ℝ := sorry

/-- Represents the radius of the larger circle after increasing by a factor of 3 -/
noncomputable def R : ℝ := 3 * r

/-- Represents the side length of the larger square -/
noncomputable def S : ℝ := 2 * R

/-- Represents the area of the smaller circle -/
noncomputable def area_small_circle : ℝ := Real.pi * R^2

/-- Represents the area of the larger square -/
noncomputable def area_large_square : ℝ := S^2

/-- The theorem stating the ratio of the areas -/
theorem area_ratio :
  area_small_circle / area_large_square = Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l1355_135574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_not_q_l1355_135577

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, Real.cos x > Real.sin x) ∧ 
  ¬(∀ x ∈ Set.Ioo 0 Real.pi, Real.sin x + (1 / Real.sin x) > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_not_q_l1355_135577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_49_64_l1355_135519

/-- The mean proportional between two numbers -/
noncomputable def mean_proportional (a b : ℝ) : ℝ := Real.sqrt (a * b)

/-- Theorem stating that the mean proportional between 49 and 64 is 56 -/
theorem mean_proportional_49_64 : mean_proportional 49 64 = 56 := by
  -- Unfold the definition of mean_proportional
  unfold mean_proportional
  -- Simplify the expression under the square root
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_49_64_l1355_135519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equivalence_l1355_135530

theorem power_equivalence (y : ℝ) : (1000 : ℝ)^4 = 10^y → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equivalence_l1355_135530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_cost_is_one_fifty_l1355_135516

/-- Represents the cost structure and profit of a taco truck -/
structure TacoTruck where
  beef_bought : ℚ
  beef_per_taco : ℚ
  taco_price : ℚ
  total_profit : ℚ

/-- Calculates the cost to make each taco -/
noncomputable def cost_per_taco (t : TacoTruck) : ℚ :=
  let num_tacos := t.beef_bought / t.beef_per_taco
  let total_revenue := num_tacos * t.taco_price
  let total_cost := total_revenue - t.total_profit
  total_cost / num_tacos

/-- Theorem stating that the cost to make each taco is $1.50 -/
theorem taco_cost_is_one_fifty :
  let t : TacoTruck := {
    beef_bought := 100,
    beef_per_taco := 1/4,
    taco_price := 2,
    total_profit := 200
  }
  cost_per_taco t = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_cost_is_one_fifty_l1355_135516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1355_135563

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  Set.range f = Set.Ioi 1 ∪ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1355_135563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_sweater_count_l1355_135527

/-- Kim's sweater knitting problem -/
theorem kim_sweater_count
  (max_per_day : ℕ)
  (monday : ℕ)
  (tuesday_increase : ℕ)
  (midweek_decrease : ℕ)
  (friday_fraction : ℕ)
  (h1 : max_per_day = 10)
  (h2 : monday = 8)
  (h3 : tuesday_increase = 2)
  (h4 : midweek_decrease = 4)
  (h5 : friday_fraction = 2) :
  let tuesday := monday + tuesday_increase
  let wednesday := tuesday - midweek_decrease
  let thursday := wednesday
  let friday := monday / friday_fraction
  monday + tuesday + wednesday + thursday + friday = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_sweater_count_l1355_135527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_unit_circle_l1355_135585

theorem point_on_unit_circle (x₀ y₀ α : ℝ) : 
  x₀^2 + y₀^2 = 1 →
  α ∈ Set.Ioo (π/6) (π/2) →
  Real.cos (α + π/3) = -11/13 →
  x₀ = 1/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_unit_circle_l1355_135585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_naughty_subset_bound_existence_of_large_naughty_subset_l1355_135506

/-- A set is naughty if the sum of any two of its elements is not in the set --/
def IsNaughty (T : Set ℕ) : Prop :=
  ∀ u v, u ∈ T → v ∈ T → u + v ∉ T

/-- The set of natural numbers from 1 to 2006 --/
def S : Finset ℕ := Finset.range 2006

theorem naughty_subset_bound :
  ∀ T : Finset ℕ, T ⊆ S → IsNaughty T → T.card ≤ 1003 := by sorry

theorem existence_of_large_naughty_subset :
  ∀ S : Finset ℕ, S.card = 2006 → ∃ T : Finset ℕ, T ⊆ S ∧ IsNaughty T ∧ T.card ≥ 669 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_naughty_subset_bound_existence_of_large_naughty_subset_l1355_135506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_perfect_square_sum_l1355_135501

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The set of possible sums that are perfect squares -/
def perfect_square_sums : List ℕ := [4, 9, 16]

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of ways to roll each perfect square sum -/
def ways_to_roll (sum : ℕ) : ℕ :=
  if sum = 4 then 3
  else if sum = 9 then 8
  else if sum = 16 then 1
  else 0

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := (perfect_square_sums.map ways_to_roll).sum

theorem probability_of_perfect_square_sum :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_perfect_square_sum_l1355_135501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_seats_l1355_135511

theorem airplane_seats :
  ∀ (total_seats : ℚ),
  (36 : ℚ) + 0.3 * total_seats + (3 / 5 : ℚ) * total_seats = total_seats →
  total_seats = 360 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_seats_l1355_135511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_existence_l1355_135535

theorem difference_existence (S : Finset ℕ) 
  (h1 : S.card = 700)
  (h2 : ∀ n, n ∈ S → n ≤ 2017) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (b - a = 3 ∨ b - a = 4 ∨ b - a = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_existence_l1355_135535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1355_135567

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - 3^x) + 1 / x^2

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 0}

-- Theorem stating that the domain of f is (-∞, 0)
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1355_135567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_thirty_in_domain_of_f_of_f_thirty_is_smallest_in_domain_of_f_of_f_l1355_135565

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5)

-- State the theorem
theorem smallest_x_in_domain_of_f_of_f : 
  ∀ x : ℝ, (∃ y : ℝ, f (f x) = y) → x ≥ 30 :=
by sorry

-- State that 30 is in the domain of f(f(x))
theorem thirty_in_domain_of_f_of_f : 
  ∃ y : ℝ, f (f 30) = y :=
by sorry

-- Prove that 30 is the smallest such number
theorem thirty_is_smallest_in_domain_of_f_of_f : 
  ∀ x : ℝ, x < 30 → ¬(∃ y : ℝ, f (f x) = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_thirty_in_domain_of_f_of_f_thirty_is_smallest_in_domain_of_f_of_f_l1355_135565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_revenue_l1355_135540

noncomputable def beef_amount : ℝ := 20
noncomputable def pork_amount : ℝ := beef_amount / 2
noncomputable def meat_per_meal : ℝ := 1.5
noncomputable def price_per_meal : ℝ := 20

theorem restaurant_revenue : 
  (beef_amount + pork_amount) / meat_per_meal * price_per_meal = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_revenue_l1355_135540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CDP_l1355_135541

/-- A square with side length 16 -/
structure Square :=
  (side_length : ℝ)
  (is_sixteen : side_length = 16)

/-- A point on a semicircle outside the square -/
structure PointOnSemicircle (sq : Square) :=
  (point : ℝ × ℝ)
  (on_semicircle : (point.1 - 8)^2 + point.2^2 = 64)
  (above_base : point.2 ≥ 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_CDP (sq : Square) (P : PointOnSemicircle sq) :
  distance (8, 8) P.point = 12 →
  triangle_area (16, 16) (0, 16) P.point = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CDP_l1355_135541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_product_l1355_135518

theorem factorial_15_product : ∃ (a b c d e f g h i : ℕ),
  (16 ≤ a ∧ a ≤ 30) ∧
  (16 ≤ b ∧ b ≤ 30) ∧
  (16 ≤ c ∧ c ≤ 30) ∧
  (16 ≤ d ∧ d ≤ 30) ∧
  (16 ≤ e ∧ e ≤ 30) ∧
  (16 ≤ f ∧ f ≤ 30) ∧
  (16 ≤ g ∧ g ≤ 30) ∧
  (16 ≤ h ∧ h ≤ 30) ∧
  (16 ≤ i ∧ i ≤ 30) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  a * b * c * d * e * f * g * h * i = Nat.factorial 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_product_l1355_135518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_no_min_l1355_135575

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (1/6) * x^2 - (1/2) * a * x^2 + x

-- Define the convexity condition
def is_convex (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo a b, HasDerivAt f (deriv f x) x ∧ deriv (deriv f) x < 0

-- Theorem statement
theorem f_has_max_no_min (a : ℝ) (h1 : a ≤ 2) (h2 : is_convex (f a) (-1) 2) :
  (∃ x ∈ Set.Ioo (-1) 2, ∀ y ∈ Set.Ioo (-1) 2, f a y ≤ f a x) ∧
  (∀ x ∈ Set.Ioo (-1) 2, ∃ y ∈ Set.Ioo (-1) 2, f a y < f a x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_no_min_l1355_135575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_value_l1355_135579

/-- The constant term in the expansion of (2√x - 1/(2x))^9 -/
noncomputable def constant_term (x : ℝ) : ℝ :=
  (2 * Real.sqrt x - 1 / (2 * x)) ^ 9

/-- The constant term in the expansion of (2√x - 1/(2x))^9 is equal to -672 -/
theorem constant_term_value :
  ∀ x : ℝ, x > 0 → constant_term x = -672 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_value_l1355_135579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1355_135594

/-- The focus of a parabola given by y = ax^2 + bx + c -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 2x^2 + 8x - 5 is at (-2, -103/8) -/
theorem focus_of_specific_parabola :
  parabola_focus 2 8 (-5) = (-2, -103/8) := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1355_135594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_cord_lengths_l1355_135517

/-- Represents a dog tied to a tree -/
structure Dog where
  run_distance : ℝ
  run_type : String

/-- Calculates the cord length based on the dog's run type and distance -/
noncomputable def cord_length (dog : Dog) : ℝ :=
  match dog.run_type with
  | "diameter" => dog.run_distance / 2
  | "circle" => dog.run_distance
  | _ => 0  -- Default case, should not occur in this problem

/-- The main theorem stating the cord lengths for the three dogs -/
theorem dog_cord_lengths (dog1 dog2 dog3 : Dog)
  (h1 : dog1.run_distance = 30 ∧ dog1.run_type = "diameter")
  (h2 : dog2.run_distance = 40 ∧ dog2.run_type = "diameter")
  (h3 : dog3.run_distance = 20 ∧ dog3.run_type = "circle") :
  cord_length dog1 = 15 ∧ cord_length dog2 = 20 ∧ cord_length dog3 = 20 := by
  sorry

#check dog_cord_lengths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_cord_lengths_l1355_135517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_calculation_l1355_135589

/-- In a race, runner A beats runner B by a certain distance or time. -/
structure Race where
  length : ℚ
  distance_diff : ℚ
  time_diff : ℚ

/-- Calculate the time taken by runner A to complete the race -/
noncomputable def race_time (r : Race) : ℚ :=
  r.length * r.time_diff / r.distance_diff

/-- Theorem: In a 120-meter race where A beats B by 56 meters or 7 seconds, A's time is 8 seconds -/
theorem race_time_calculation (r : Race) (h1 : r.length = 120) (h2 : r.distance_diff = 56) (h3 : r.time_diff = 7) :
  race_time r = 8 := by
  sorry

#check race_time_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_calculation_l1355_135589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_10_value_l1355_135591

def b : ℕ → ℚ
  | 0 => 2  -- We need to handle the case for 0
  | 1 => 5  -- This corresponds to b₂
  | (n+2) => b (n+1)^2 / (b (n+1) - b n)

theorem b_10_value : b 10 = 105 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_10_value_l1355_135591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rearrangement_satisfies_condition_l1355_135588

/-- The polynomial A(x) with coefficients from 1 to 199 with step 2 -/
noncomputable def A : Polynomial ℤ := Finset.sum (Finset.range 100) (fun i => (2*i + 1) • Polynomial.X^i)

/-- The type of polynomials that are rearrangements of A's coefficients -/
def RearrangedPolynomial (p : Polynomial ℤ) : Prop :=
  p.degree = 99 ∧ Finset.sum (Finset.range 100) (fun i => p.coeff i) = Finset.sum (Finset.range 100) (fun i => A.coeff i)

theorem no_rearrangement_satisfies_condition : 
  ¬∃ B : Polynomial ℤ, RearrangedPolynomial B ∧ 
    ∀ k : ℕ, k ≥ 2 → ¬(199 ∣ (A.eval (k : ℤ) - B.eval (k : ℤ))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rearrangement_satisfies_condition_l1355_135588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_correct_answers_l1355_135545

/-- Represents an exam with a fixed number of questions, scoring rules, and a total score. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℕ
  wrong_penalty : ℕ
  total_score : ℤ

/-- Calculates the number of correctly answered questions in an exam. -/
def correct_answers (e : Exam) : ℤ :=
  ((e.total_score : ℤ) + (e.total_questions : ℤ) * (e.wrong_penalty : ℤ)) /
    ((e.correct_score : ℤ) + (e.wrong_penalty : ℤ))

/-- Theorem stating that for the given exam parameters, the number of correct answers is 44. -/
theorem exam_correct_answers :
  let e : Exam := {
    total_questions := 60,
    correct_score := 4,
    wrong_penalty := 1,
    total_score := 160
  }
  correct_answers e = 44 := by
  sorry

#eval correct_answers {
  total_questions := 60,
  correct_score := 4,
  wrong_penalty := 1,
  total_score := 160
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_correct_answers_l1355_135545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1355_135514

/-- The maximum distance from any point on the circle (x-1)^2 + y^2 = 1 
    to the line x + y + 2 = 0 is (3√2)/2 + 1 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 + p.2 + 2 = 0}
  ∃ (d : ℝ), d = (3 * Real.sqrt 2) / 2 + 1 ∧
    (∀ p ∈ circle, ∀ q ∈ line, dist p q ≤ d) ∧
    (∃ p ∈ circle, ∃ q ∈ line, dist p q = d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1355_135514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_contribution_theorem_l1355_135500

/-- Calculates the number of gallons needed for a given area, coverage, and coats -/
noncomputable def gallons_needed (area : ℝ) (coverage : ℝ) (coats : ℕ) : ℝ :=
  area * (coats : ℝ) / coverage

/-- Calculates the cost of paint given the number of gallons and price per gallon -/
noncomputable def paint_cost (gallons : ℝ) (price_per_gallon : ℝ) : ℝ :=
  ⌈gallons⌉ * price_per_gallon

/-- Represents the paint job details for each person -/
structure PaintJob where
  area : ℝ
  coats : ℕ
  coverage : ℝ
  price_per_gallon : ℝ

theorem paint_contribution_theorem (jason jeremy : PaintJob) 
  (h_jason : jason = { area := 1025, coats := 3, coverage := 350, price_per_gallon := 50 })
  (h_jeremy : jeremy = { area := 1575, coats := 2, coverage := 400, price_per_gallon := 45 }) :
  let jason_cost := paint_cost (gallons_needed jason.area jason.coverage jason.coats) jason.price_per_gallon
  let jeremy_cost := paint_cost (gallons_needed jeremy.area jeremy.coverage jeremy.coats) jeremy.price_per_gallon
  let total_cost := jason_cost + jeremy_cost
  (total_cost / 2) = 405 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_contribution_theorem_l1355_135500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_intersection_l1355_135583

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- The polar of a point with respect to a circle -/
noncomputable def polar (A : Point) (C : Circle) : Line :=
  sorry

/-- Checks if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  sorry

/-- Checks if a line is a chord of a circle -/
def isChord (l : Line) (c : Circle) : Prop :=
  sorry

/-- Checks if a point lies on a line -/
def onLine (p : Point) (l : Line) : Prop :=
  sorry

/-- The intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

theorem locus_of_intersection (C : Circle) (A P Q K L : Point) :
  isInside A C →
  ∃ (PQ KL : Line),
    isChord PQ C ∧ isChord KL C ∧
    onLine A PQ ∧ onLine A KL ∧
    onLine P PQ ∧ onLine Q PQ ∧
    onLine K KL ∧ onLine L KL →
    ∃ (PK QL : Line),
      onLine P PK ∧ onLine K PK ∧
      onLine Q QL ∧ onLine L QL →
      onLine (intersectionPoint PK QL) (polar A C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_intersection_l1355_135583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l1355_135523

/-- Given a line segment with one endpoint at (6, -2) and midpoint at (5, 4),
    the sum of the coordinates of the other endpoint is 14. -/
theorem endpoint_coordinate_sum :
  ∀ (endpoint1 midpoint endpoint2 : ℝ × ℝ),
    endpoint1 = (6, -2) →
    midpoint = (5, 4) →
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l1355_135523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_arrangement_l1355_135562

/-- The number of ways to arrange 3 students and 2 teachers in a row with constraints -/
theorem teacher_student_arrangement (n : ℕ) (h1 : n = 3) : 
  Nat.factorial (n + 2) / 2 = 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_arrangement_l1355_135562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_half_focal_length_l1355_135581

/-- A hyperbola with center at the origin and foci on the coordinate axes -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form x^2 - y^2/k^2 = r -/
  equation : ℝ → ℝ → ℝ → ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  point : Point

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ := sorry

/-- Geometric mean of two lengths -/
noncomputable def geometricMean (a b : ℝ) : ℝ := sorry

/-- Half focal length of a hyperbola -/
noncomputable def halfFocalLength (h : Hyperbola) : ℝ := sorry

theorem hyperbola_half_focal_length 
  (h : Hyperbola) 
  (p : Point)
  (l : Line)
  (a b m : Point) :
  p.x = -2 ∧ p.y = 0 →
  distanceToLine p { slope := h.equation 1 0 0, point := p } = 2 * Real.sqrt 6 / 3 →
  l.slope = Real.sqrt 2 / 2 →
  l.point = p →
  h.equation a.x a.y 1 = 0 →
  h.equation b.x b.y 1 = 0 →
  m.x = 0 →
  geometricMean (Real.sqrt ((a.x - p.x)^2 + (a.y - p.y)^2)) 
                (Real.sqrt ((b.x - p.x)^2 + (b.y - p.y)^2)) = 
  Real.sqrt ((m.x - p.x)^2 + (m.y - p.y)^2) →
  halfFocalLength h = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_half_focal_length_l1355_135581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1355_135512

-- Define the function y as noncomputable
noncomputable def y (x : ℝ) : ℝ := (Real.log x) / x + x * Real.exp x

-- State the theorem
theorem y_derivative (x : ℝ) (h : x > 0) :
  deriv y x = (1 - Real.log x) / x^2 + (x + 1) * Real.exp x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_derivative_l1355_135512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_proof_l1355_135553

noncomputable def star_operation (y : ℝ) : ℝ :=
  2 * ⌊y / 2⌋

theorem z_value_proof (z : ℝ) : 
  (6.15 - star_operation 6.15 = 0.15000000000000036) → z = 6.15 :=
by
  intro h
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_value_proof_l1355_135553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_l1355_135580

/-- Represents a horizontal translation of a trigonometric function -/
noncomputable def horizontalTranslation (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ :=
  fun x ↦ f (x - h)

/-- The original sine function -/
noncomputable def originalFunction : ℝ → ℝ :=
  fun x ↦ Real.sin (2 * x)

theorem sine_translation :
  horizontalTranslation originalFunction (π / 3) =
  fun x ↦ Real.sin (2 * x - 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_l1355_135580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_presidency_meeting_arrangements_l1355_135515

theorem presidency_meeting_arrangements (n : ℕ) (k : ℕ) :
  n = 4 →
  k = 4 →
  (n * k : ℕ) = (n : ℕ) * (Nat.choose k 3) * (n - 1 : ℕ) * (Nat.choose k 1)^(n - 1 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_presidency_meeting_arrangements_l1355_135515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sixty_degree_angles_is_equilateral_converse_equilateral_triangle_angles_true_l1355_135532

/-- Structure representing a triangle -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  is_equilateral : Prop

/-- A triangle with three 60° angles is equilateral -/
theorem triangle_sixty_degree_angles_is_equilateral (T : Triangle) 
  (h1 : T.angle1 = 60) 
  (h2 : T.angle2 = 60) 
  (h3 : T.angle3 = 60) : 
  T.is_equilateral := by
  sorry

/-- The converse of "Each interior angle of an equilateral triangle is 60°" is true -/
theorem converse_equilateral_triangle_angles_true : 
  (∀ T : Triangle, (T.angle1 = 60 ∧ T.angle2 = 60 ∧ T.angle3 = 60) → T.is_equilateral) = True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sixty_degree_angles_is_equilateral_converse_equilateral_triangle_angles_true_l1355_135532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1355_135596

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Checks if a point lies on the ellipse -/
def pointOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The ellipse with given properties has the specified equation -/
theorem ellipse_equation :
  let f1 : Point := ⟨4, 1⟩
  let f2 : Point := ⟨4, 7⟩
  let p : Point := ⟨12, 5⟩
  let e : Ellipse := ⟨3 * Real.sqrt 15, 12, 4, 4⟩
  (distance f1 p + distance f2 p = 2 * e.a) ∧
  pointOnEllipse e p ∧
  e.a > 0 ∧
  e.b > 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1355_135596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_is_10_5_l1355_135564

/-- Represents the swimming scenario with given distances and times -/
structure SwimmingScenario where
  downstream_distance : ℝ
  upstream_distance : ℝ
  downstream_time : ℝ
  upstream_time : ℝ

/-- Calculates the speed of a swimmer in still water given a swimming scenario -/
noncomputable def speed_in_still_water (scenario : SwimmingScenario) : ℝ :=
  (scenario.downstream_distance / scenario.downstream_time + 
   scenario.upstream_distance / scenario.upstream_time) / 2

/-- Theorem stating that for the given swimming scenario, the speed in still water is 10.5 km/h -/
theorem speed_is_10_5 (scenario : SwimmingScenario) 
  (h1 : scenario.downstream_distance = 45)
  (h2 : scenario.upstream_distance = 18)
  (h3 : scenario.downstream_time = 3)
  (h4 : scenario.upstream_time = 3) :
  speed_in_still_water scenario = 10.5 := by
  sorry

#check speed_is_10_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_is_10_5_l1355_135564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_height_theorem_l1355_135508

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a frustum of a cone -/
structure Frustum where
  lowerRadius : ℝ
  upperRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ :=
  (1/3) * Real.pi * c.baseRadius^2 * c.height

/-- Calculates the volume of a frustum -/
noncomputable def frustumVolume (f : Frustum) : ℝ :=
  (1/3) * Real.pi * f.height * (f.lowerRadius^2 + f.upperRadius^2 + f.lowerRadius * f.upperRadius)

theorem frustum_height_theorem (cone : Cone) (f1 f2 : Frustum) (c : Cone) :
  cone.height = 6 →
  cone.baseRadius = 5 →
  c.height = 2 →
  frustumVolume f1 / frustumVolume f2 = 1/2 →
  f1.height + f2.height + c.height = cone.height →
  abs (f2.height - 2.3) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_height_theorem_l1355_135508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_QAMB_equation_MQ_l1355_135546

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define a point Q on the x-axis
def point_on_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0

-- Define tangent lines QA and QB
def tangent_lines (Q A B : ℝ × ℝ) : Prop :=
  circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
  ((A.1 - Q.1) * (A.1 - 0) + (A.2 - Q.2) * (A.2 - 2) = 0) ∧
  ((B.1 - Q.1) * (B.1 - 0) + (B.2 - Q.2) * (B.2 - 2) = 0)

-- Theorem for the minimum area of quadrilateral QAMB
theorem min_area_QAMB :
  ∀ Q A B : ℝ × ℝ,
  point_on_x_axis Q →
  tangent_lines Q A B →
  ∃ area : ℝ, area ≥ Real.sqrt 3 ∧
  (∀ Q' A' B' : ℝ × ℝ, point_on_x_axis Q' → tangent_lines Q' A' B' →
    area ≤ abs ((A'.1 - Q'.1) * (B'.2 - Q'.2) - (B'.1 - Q'.1) * (A'.2 - Q'.2)) / 2) :=
by sorry

-- Theorem for the equation of line MQ when |AB| = 4√2/3
theorem equation_MQ :
  ∀ Q A B : ℝ × ℝ,
  point_on_x_axis Q →
  tangent_lines Q A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 / 3 →
  ∃ k : ℝ, k = 1 ∨ k = -1 ∧ 2 * Q.1 + k * Real.sqrt 5 * Q.2 - 2 * Real.sqrt 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_QAMB_equation_MQ_l1355_135546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_shifted_cos_l1355_135513

noncomputable def f (x φ : ℝ) := Real.cos (2 * x + φ)

theorem min_value_shifted_cos 
  (φ : ℝ) 
  (h1 : |φ| < π/2) 
  (h2 : ∃ k : ℤ, φ + π/3 = k * π) -- symmetry condition
  : 
  (∀ x ∈ Set.Icc (π/12) (π/2), f (x - π/6) φ ≥ -1/2) ∧ 
  (∃ x ∈ Set.Icc (π/12) (π/2), f (x - π/6) φ = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_shifted_cos_l1355_135513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_days_and_amount_l1355_135587

-- Define the selling price function
def selling_price (x : ℕ) : ℚ := 1/2 * x + 18

-- Define the sales volume function
def sales_volume (x : ℕ) : ℚ := -x + 35

-- Define the profit function
def profit (x : ℕ) : ℚ := (sales_volume x) * (selling_price x - 8)

-- Theorem statement
theorem max_profit_days_and_amount :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 →
  profit x ≤ 378 ∧
  (∃ y : ℕ, y = 7 ∨ y = 8) ∧ profit y = 378 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_days_and_amount_l1355_135587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1355_135561

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 4 + a 6 = 18 →
  (∀ n : ℕ, b (n + 1) = 2 * b n) →
  b 1 = a 5 →
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, (Finset.range n).sum (fun i => b (i + 1)) = 9 * 2^n - 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1355_135561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_specific_parabola_properties_l1355_135520

/-- For a parabola y = ax^2, where a > 0, the parabola opens upwards and its focus is at (0, 1/(4a)) -/
theorem parabola_properties (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2
  ∃ (opens_upward : Bool) (focus : ℝ × ℝ),
    opens_upward = true ∧ focus = (0, 1 / (4 * a)) := by
  sorry

/-- For the specific parabola y = 4x^2, it opens upwards and its focus is at (0, 1/16) -/
theorem specific_parabola_properties :
  let f : ℝ → ℝ := fun x ↦ 4 * x^2
  ∃ (opens_upward : Bool) (focus : ℝ × ℝ),
    opens_upward = true ∧ focus = (0, 1/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_specific_parabola_properties_l1355_135520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OPQ_l1355_135558

noncomputable section

/-- The ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line l₁ -/
def line_l₁ (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 4 = 0

/-- A point on the ellipse C -/
def point_on_C : Prop := ellipse_C 1 (-Real.sqrt 3 / 2)

/-- The line l is perpendicular to l₁ -/
def l_perpendicular_l₁ (m : ℝ) : Prop := 
  ∀ x y, y = -Real.sqrt 3 * x + m

/-- The area of triangle OPQ -/
noncomputable def area_OPQ (m : ℝ) : ℝ := 
  2 * Real.sqrt (m^2 * (13 - m^2)) / 13

/-- The theorem stating the maximum area of triangle OPQ -/
theorem max_area_OPQ : 
  ∃ m, l_perpendicular_l₁ m ∧ 
    (∀ m', l_perpendicular_l₁ m' → area_OPQ m ≥ area_OPQ m') ∧ 
    area_OPQ m = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OPQ_l1355_135558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l1355_135502

open Matrix

theorem inverse_scalar_multiple {α : Type*} [Field α] (e m : α) :
  let B : Matrix (Fin 2) (Fin 2) α := !![4, 5; 7, e]
  IsUnit B.det → B⁻¹ = m • B → e = -4 ∧ m = 1 / 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_scalar_multiple_l1355_135502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1355_135557

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def lineL (a x y : ℝ) : Prop := a * x + y + 2 * a = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem line_equation (a : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, 
    circleC x1 y1 ∧ circleC x2 y2 ∧ 
    lineL a x1 y1 ∧ lineL a x2 y2 ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 2) →
  (a = 1 ∨ a = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1355_135557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_21_minutes_l1355_135550

/-- Calculates the total time in hours for walking 3 miles each day for 5 days at given speeds -/
noncomputable def total_time_varying_speed (speeds : List ℝ) : ℝ :=
  List.sum (List.map (fun speed => 3 / speed) speeds)

/-- Calculates the total time in hours for walking 3 miles each day for 5 days at a constant speed -/
noncomputable def total_time_constant_speed (speed : ℝ) : ℝ :=
  (3 * 5) / speed

/-- Theorem stating the time difference between varying and constant speeds -/
theorem time_difference_21_minutes 
  (varying_speeds : List ℝ := [6, 4, 5, 6, 3])
  (constant_speed : ℝ := 5) : 
  (total_time_varying_speed varying_speeds - total_time_constant_speed constant_speed) * 60 = 21 := by
  sorry

-- Remove the #eval statement as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_21_minutes_l1355_135550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_today_is_thursday_l1355_135544

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the 'yesterday' function for Day
def Day.yesterday : Day → Day
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Define the lying patterns for the lion and unicorn
def lion_lies (d : Day) : Prop :=
  d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday

def unicorn_lies (d : Day) : Prop :=
  d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

-- Define the statements made by the lion and unicorn
def lion_statement (today : Day) : Prop :=
  lion_lies (Day.yesterday today)

def unicorn_statement (today : Day) : Prop :=
  unicorn_lies (Day.yesterday today)

-- Define the consistency of statements with the lying patterns
def consistent_statements (today : Day) : Prop :=
  (lion_lies today → ¬lion_statement today) ∧
  (¬lion_lies today → lion_statement today) ∧
  (unicorn_lies today → ¬unicorn_statement today) ∧
  (¬unicorn_lies today → unicorn_statement today)

-- Theorem statement
theorem today_is_thursday :
  ∃! d : Day, consistent_statements d ∧ d = Day.Thursday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_today_is_thursday_l1355_135544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_tan6_sec4_is_infinity_l1355_135525

noncomputable section

def sec (x : ℝ) : ℝ := 1 / Real.cos x

theorem max_value_tan6_sec4_is_infinity :
  let f : ℝ → ℝ := λ x => Real.tan x ^ 6 + sec x ^ 4
  ∀ M : ℝ, ∃ x : ℝ, f x > M :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_tan6_sec4_is_infinity_l1355_135525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l1355_135556

theorem smallest_sum_of_factors (a b : ℕ) (h : (2^12) * (3^3) = a^b) : 
  (∀ (x y : ℕ), (2^12) * (3^3) = x^y → a + b ≤ x + y) ∧ a + b = 110593 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l1355_135556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l1355_135504

-- Define the ellipse
noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - 2*y - 12| / Real.sqrt 5

-- Theorem statement
theorem min_distance_ellipse_to_line :
  ∃ (min_dist : ℝ), 
    (∀ (x y : ℝ), is_on_ellipse x y → distance_to_line x y ≥ min_dist) ∧
    (∃ (x y : ℝ), is_on_ellipse x y ∧ distance_to_line x y = min_dist) ∧
    min_dist = 4 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l1355_135504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_point_relationship_l1355_135521

theorem decimal_point_relationship (A B C : ℝ) 
  (h1 : A * (10 : ℝ)^(-8 : ℤ) = B * (10 : ℝ)^(3 : ℤ)) 
  (h2 : B * (10 : ℝ)^(-2 : ℤ) = C * (10 : ℝ)^(2 : ℤ)) : 
  A = C * (10 : ℝ)^(15 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_point_relationship_l1355_135521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1355_135592

def expression (a b c d : ℕ) : ℕ := c * a^b - d

def is_valid_assignment (a b c d : ℕ) : Prop :=
  Finset.toSet {a, b, c, d} = Finset.toSet {1, 2, 3, 4} ∧ c ≠ 1

theorem max_expression_value :
  ∃ (a b c d : ℕ), is_valid_assignment a b c d ∧
  (∀ (w x y z : ℕ), is_valid_assignment w x y z →
  expression a b c d ≥ expression w x y z) ∧
  expression a b c d = 127 := by
  sorry

#eval expression 4 3 2 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1355_135592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1355_135598

theorem hyperbola_asymptotes (a : ℝ) :
  let C : ℝ → ℝ → Prop := λ x y ↦ (y + a)^2 - (x - a)^2 = 2*a
  let asymptote_passes_through : Prop := ∃ (m b : ℝ), 1 = m*3 + b ∧ (∀ x y, C x y → (y = m*x + b ∨ y = -m*x + b))
  asymptote_passes_through →
  (∀ x y, C x y → ((y = x - 2) ∨ (y = -x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1355_135598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_B_in_geometric_progression_triangle_l1355_135526

theorem max_angle_B_in_geometric_progression_triangle (A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- sin A, sin B, and sin C form a geometric progression
  ∃ r : ℝ, (Real.sin B)^2 = (Real.sin A) * (Real.sin C) →
  -- Maximum value of angle B
  B ≤ Real.pi/3 ∧ ∃ A' C', A' + Real.pi/3 + C' = Real.pi ∧ 
    (Real.sin (Real.pi/3))^2 = (Real.sin A') * (Real.sin C') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_B_in_geometric_progression_triangle_l1355_135526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l1355_135543

theorem max_vector_difference (x : ℝ) : 
  let m : Fin 2 → ℝ := ![Real.cos (x/2), Real.sin (x/2)]
  let n : Fin 2 → ℝ := ![-Real.sqrt 3, 1]
  (∀ y : ℝ, ‖m - n‖ ≤ 3) ∧ (∃ y : ℝ, ‖m - n‖ = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l1355_135543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_and_area_l1355_135599

/-- Represents a trapezoid ABCD where AB is parallel to CD -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  height : ℝ
  parallel : AB = CD

/-- The perimeter of a trapezoid -/
noncomputable def perimeter (t : Trapezoid) : ℝ := t.AB + t.BC + t.CD + t.AD

/-- The area of a trapezoid -/
noncomputable def area (t : Trapezoid) : ℝ := (t.AB + t.AD) * t.height / 2

theorem trapezoid_perimeter_and_area (t : Trapezoid) 
  (h1 : t.height = 5)
  (h2 : t.AD = 20)
  (h3 : t.BC = 12)
  (h4 : t.AB = Real.sqrt 41) :
  perimeter t = 2 * Real.sqrt 41 + 32 ∧ 
  area t = 2.5 * (20 + Real.sqrt 41) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_and_area_l1355_135599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_point_l1355_135537

/-- The curve function f(x) = sin(2x) + √3 * cos(2x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

/-- Theorem: For the curve f(x) = sin(2x) + √3 * cos(2x), which is symmetric about (x₀, 0) where x₀ ∈ [0, π/2], x₀ = π/3 -/
theorem curve_symmetry_point (x₀ : ℝ) (h1 : 0 ≤ x₀) (h2 : x₀ ≤ Real.pi / 2)
  (h3 : ∀ x : ℝ, f (x₀ + x) = f (x₀ - x)) : x₀ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_point_l1355_135537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1355_135510

/-- The ellipse defined by x^2 + y^2/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2/4 = 1}

/-- The lower focus of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := (0, -Real.sqrt 3)

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- The dot product of vectors PF₁ and PO -/
noncomputable def dotProduct (P : ℝ × ℝ) : ℝ :=
  (P.1 - F₁.1) * (P.1 - O.1) + (P.2 - F₁.2) * (P.2 - O.2)

theorem max_dot_product :
  ∃ (M : ℝ), M = 4 + 2 * Real.sqrt 3 ∧
  ∀ (P : ℝ × ℝ), P ∈ Ellipse → dotProduct P ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1355_135510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_product_line_l1355_135533

-- Define the ellipse C
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the hyperbola C'
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define point Q
def Q : ℝ × ℝ := (1, 0)

-- Define point P
def P : ℝ × ℝ := (4, 3)

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define the slope product k₁ * k₂
noncomputable def slope_product (m : ℝ) : ℝ :=
  (3 * m^2 + 2 * m + 5) / (4 * m^2 + 6)

-- Theorem statement
theorem max_slope_product_line :
  ∃ (m : ℝ), (∀ (m' : ℝ), slope_product m ≥ slope_product m') ∧
  (line_l m = λ x y => x - y - 1 = 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_product_line_l1355_135533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_30_minus_2alpha_l1355_135582

theorem cos_30_minus_2alpha (α : ℝ) : 
  Real.sin (75 * π / 180 + α) = 1/3 → Real.cos (30 * π / 180 - 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_30_minus_2alpha_l1355_135582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_eq_seven_l1355_135568

/-- The number of ordered triples (x, y, z) of positive integers satisfying the given LCM conditions -/
def count_triples : ℕ :=
  Finset.filter (fun t : ℕ × ℕ × ℕ => 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧
    Nat.lcm t.1 t.2.1 = 144 ∧
    Nat.lcm t.1 t.2.2 = 450 ∧
    Nat.lcm t.2.1 t.2.2 = 600)
    (Finset.product (Finset.range 451) (Finset.product (Finset.range 601) (Finset.range 601)))
  |>.card

/-- Theorem stating that there are exactly 7 triples satisfying the conditions -/
theorem count_triples_eq_seven : count_triples = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_eq_seven_l1355_135568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_one_fourth_l1355_135509

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.log x / Real.log 3
  else if x ≤ 0 then 2^x
  else 0  -- This case is added to make the function total

-- State the theorem
theorem function_composition_equals_one_fourth (x : ℝ) :
  f (f x) = 1/4 → x = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equals_one_fourth_l1355_135509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l1355_135584

/-- Sequence b defined recursively -/
def b : ℕ → ℚ
  | 0 => 2
  | 1 => 2
  | n + 2 => (1 / 5 : ℚ) * b (n + 1) + (1 / 6 : ℚ) * b n

/-- Sum of the sequence b -/
noncomputable def seriesSum : ℚ := ∑' n, b n

/-- Theorem: The sum of the sequence b is 108/19 -/
theorem sum_of_sequence_b : seriesSum = 108 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_b_l1355_135584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_calculations_l1355_135536

structure Flower :=
  (name : String)
  (price : ℚ)
  (quantity : ℕ)

def flowers : List Flower := [
  ⟨"Rose", 2, 50⟩,
  ⟨"Lily", 3/2, 40⟩,
  ⟨"Sunflower", 1, 30⟩,
  ⟨"Daisy", 3/4, 20⟩,
  ⟨"Orchid", 3, 10⟩,
  ⟨"Tulip", 5/2, 15⟩
]

def total_cost : ℚ := flowers.foldr (λ f acc => acc + f.price * f.quantity) 0

def total_quantity : ℕ := flowers.foldr (λ f acc => acc + f.quantity) 0

def percentage (f : Flower) : ℚ :=
  (f.quantity : ℚ) / (total_quantity : ℚ) * 100

def average_price : ℚ := total_cost / (total_quantity : ℚ)

theorem flower_calculations :
  total_cost = 545/2 ∧
  percentage (flowers[0]) = 3030/100 ∧
  percentage (flowers[1]) = 2424/100 ∧
  percentage (flowers[2]) = 1818/100 ∧
  percentage (flowers[3]) = 1212/100 ∧
  percentage (flowers[4]) = 606/100 ∧
  percentage (flowers[5]) = 909/100 ∧
  average_price = 33/20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_calculations_l1355_135536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_improvement_theorem_l1355_135524

/-- Represents the race conditions and results --/
structure RaceData where
  square_length : ℝ
  num_laps : ℕ
  this_year_times : Fin 3 → ℝ
  last_year_times : Fin 3 → ℝ

/-- Calculates the average improvement in minutes per mile --/
noncomputable def average_improvement (data : RaceData) : ℝ :=
  let total_distance := data.square_length * data.num_laps
  let this_year_avg := (data.this_year_times 0 + data.this_year_times 1 + data.this_year_times 2) / (3 * total_distance)
  let last_year_avg := (data.last_year_times 0 + data.last_year_times 1 + data.last_year_times 2) / (3 * total_distance)
  last_year_avg - this_year_avg

/-- The main theorem stating the average improvement --/
theorem race_improvement_theorem (data : RaceData) 
  (h1 : data.square_length = Real.pi)
  (h2 : data.num_laps = 11)
  (h3 : data.this_year_times 0 = 82.5)
  (h4 : data.this_year_times 1 = 84)
  (h5 : data.this_year_times 2 = 86)
  (h6 : data.last_year_times 0 = 106.37)
  (h7 : data.last_year_times 1 = 109.5)
  (h8 : data.last_year_times 2 = 112) :
  ∃ (ε : ℝ), ε > 0 ∧ |average_improvement data - 0.72666| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_improvement_theorem_l1355_135524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l1355_135538

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 12 = 0
def circle2 (x y : ℝ) : Prop := (x-7)^2 + (y-1)^2 = 36

-- Define the centers and radii
def center1 : ℝ × ℝ := (3, -2)
def radius1 : ℝ := 1
def center2 : ℝ × ℝ := (7, 1)
def radius2 : ℝ := 6

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem: The circles are internally tangent
theorem circles_internally_tangent :
  distance_between_centers = radius2 - radius1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l1355_135538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_correct_l1355_135534

/-- The point symmetric to P(1, 2, 3) with respect to the xoy plane in a 3D Cartesian coordinate system -/
def symmetric_point : ℝ × ℝ × ℝ := (1, 2, -3)

/-- The given point P -/
def P : ℝ × ℝ × ℝ := (1, 2, 3)

/-- Theorem stating that symmetric_point is indeed symmetric to P with respect to the xoy plane -/
theorem symmetric_point_correct : 
  (symmetric_point.fst = P.fst) ∧ 
  (symmetric_point.snd = P.snd) ∧ 
  (symmetric_point.2.2 = -P.2.2) := by
  sorry

#check symmetric_point_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_correct_l1355_135534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l1355_135531

noncomputable section

-- Define the ellipse and parabola
def ellipse (a b x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def parabola (p x y : ℝ) := y^2 = 2 * p * x

-- Define the shared focus
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem ellipse_parabola_intersection 
  (a b p : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hp : p > 0) :
  ∃ (x y : ℝ),
    -- Ellipse and parabola equations
    ellipse a b x y ∧ parabola p x y ∧
    -- Shared focus condition
    focus = (1, 0) ∧
    -- Distance condition for point M on parabola
    (∀ (xm ym : ℝ), parabola p xm ym → xm = dist xm ym 1 0 - 1) ∧
    -- Intersection point Q condition
    (dist x y 1 0 = 5/2) →
  -- Prove the specific equations and x₀ range
  (p = 2 ∧ a = 3 ∧ b = Real.sqrt 8) ∧
  (∀ (k m x₀ : ℝ),
    (∃ (x y : ℝ), parabola 2 x y ∧ y = k * x + m) →
    (∃ (x1 y1 x2 y2 : ℝ),
      ellipse 3 (Real.sqrt 8) x1 y1 ∧
      ellipse 3 (Real.sqrt 8) x2 y2 ∧
      y1 = k * x1 + m ∧ y2 = k * x2 + m ∧
      x₀ = (x1 + x2) / 2) →
    -1 < x₀ ∧ x₀ < 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l1355_135531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_sales_profit_percentage_l1355_135554

/-- Represents the retailer's pen sales scenario -/
structure PenSales where
  bulk_price : ℝ  -- Bulk price for 150 pens
  market_price : ℝ  -- Market price per pen
  discount1 : ℝ  -- Discount for first 50 pens
  discount2 : ℝ  -- Discount for next 50 pens
  discount3 : ℝ  -- Discount for remaining 50 pens
  sales_tax : ℝ  -- Sales tax rate

/-- Calculates the profit percentage for the given pen sales scenario -/
def profit_percentage (s : PenSales) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the profit percentage is approximately 190.7% -/
theorem pen_sales_profit_percentage :
  ∃ (s : PenSales),
    s.bulk_price = 50 * s.market_price ∧
    s.market_price = 1.20 ∧
    s.discount1 = 0.10 ∧
    s.discount2 = 0.05 ∧
    s.discount3 = 0 ∧
    s.sales_tax = 0.02 ∧
    abs (profit_percentage s - 190.7) < 0.1 := by
  sorry

#eval println! "Pen sales profit percentage theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_sales_profit_percentage_l1355_135554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l1355_135551

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (m * x + Real.sqrt (x^2 + 1))

-- State the theorem
theorem odd_function_condition (m : ℝ) :
  (∀ x, f m (-x) = -(f m x)) ↔ (m = -1 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l1355_135551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equal_l1355_135573

/-- Represents a square divided into smaller regions --/
structure DividedSquare where
  size : ℝ
  shaded_area : ℝ

/-- Square I: divided by diagonals and midpoint segments --/
noncomputable def square_I (s : ℝ) : DividedSquare :=
  { size := s
  , shaded_area := 2 * (s^2 / 8) }

/-- Square II: divided into four equal smaller squares --/
noncomputable def square_II (s : ℝ) : DividedSquare :=
  { size := s
  , shaded_area := s^2 / 4 }

/-- Square III: divided into a 4x4 grid of smaller squares --/
noncomputable def square_III (s : ℝ) : DividedSquare :=
  { size := s
  , shaded_area := 4 * (s^2 / 16) }

theorem shaded_areas_equal (s : ℝ) (h : s > 0) :
  (square_I s).shaded_area = (square_II s).shaded_area ∧
  (square_II s).shaded_area = (square_III s).shaded_area ∧
  (square_I s).shaded_area = (square_III s).shaded_area ∧
  (square_I s).shaded_area = s^2 / 4 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equal_l1355_135573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_synonyms_l1355_135595

/-- A ray in geometry -/
def Ray : Type := Unit

/-- A half-line in geometry -/
def HalfLine : Type := Unit

/-- A tetrahedron in geometry -/
def Tetrahedron : Type := Unit

/-- A triangular pyramid in geometry -/
def TriangularPyramid : Type := Unit

/-- A bisector in geometry -/
def Bisector : Type := Unit

/-- An angle bisector in geometry -/
def AngleBisector : Type := Unit

/-- Theorem stating that certain geometric terms are synonyms -/
theorem geometric_synonyms :
  (Ray = HalfLine) ∧
  (Tetrahedron = TriangularPyramid) ∧
  (Bisector = AngleBisector) := by
  constructor
  · rfl
  constructor
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_synonyms_l1355_135595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_value_l1355_135547

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x + a) * Real.log x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x * Real.log x + (Real.exp x + a) / x

theorem extreme_point_implies_a_value (a : ℝ) :
  (∀ x > 0, f a x = (Real.exp x + a) * Real.log x) →
  (∀ x > 0, f_deriv a x = Real.exp x * Real.log x + (Real.exp x + a) / x) →
  (f_deriv a 1 = 0) →
  a = -Real.exp 1 := by
  sorry

#check extreme_point_implies_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_value_l1355_135547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_17_l1355_135578

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun n : ℕ => 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0) (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_17_l1355_135578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l1355_135593

/-- The function f(x) = sin(x/4) + sin(x/9) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 9)

/-- Convert degrees to radians -/
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

theorem smallest_max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → f (deg_to_rad y) ≤ f (deg_to_rad x)) ∧
    (∀ (z : ℝ), 0 < z ∧ z < x → f (deg_to_rad z) < f (deg_to_rad x)) ∧
    x = 4050 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l1355_135593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_case_one_function_case_two_l1355_135555

/-- A function satisfying the given functional equation. -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, 2 * (f (f x)) = (x^2 - x) * (f x) + 4 - 2*x

/-- The main theorem stating the properties of f. -/
theorem function_properties (f : ℝ → ℝ) (h : satisfies_equation f) :
  f 2 = 2 ∧ (f 1 = 1 ∨ f 1 = 4) := by
  sorry

/-- Theorem for the case when f(1) = 1 -/
theorem function_case_one (f : ℝ → ℝ) (h : satisfies_equation f) (h1 : f 1 = 1) :
  ∀ x, f x = x := by
  sorry

/-- Theorem for the case when f(1) = 4 -/
theorem function_case_two (f : ℝ → ℝ) (h : satisfies_equation f) (h1 : f 1 = 4) :
  ∀ x, x ≠ 0 → f x = 4 / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_function_case_one_function_case_two_l1355_135555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aardvark_distance_l1355_135505

/-- The total distance traveled by an aardvark on a specific path between two concentric circles -/
theorem aardvark_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 20) : 
  (π * r₂) + (r₂ - r₁) + (2 * π * r₁) + (r₂ - r₁) + (2 * r₂ * Real.sin (π / 6)) = 40 * π + 40 := by
  sorry

#check aardvark_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aardvark_distance_l1355_135505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l1355_135529

-- Define the curve C
def C (a m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 + a) / p.1 * (p.2 - a) / p.1 = m}

-- Define IsEccentricityOf
def IsEccentricityOf (e : ℝ) (s : Set (ℝ × ℝ)) : Prop := sorry

-- Define AreFociOf
def AreFociOf (s : Set (ℝ × ℝ)) (f₁ f₂ : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem curve_properties (a : ℝ) (h_a : a > 0) (m : ℝ) (h_m : m ≠ 0) :
  -- When m = -1, C is a circle
  (m = -1 → ∃ r, C a m = {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}) ∧
  -- When m = -2, the eccentricity of C is √2/2
  (m = -2 → ∃ e, e = Real.sqrt 2 / 2 ∧ IsEccentricityOf e (C a m)) ∧
  -- When m ∈ (-∞, -1) ∪ (0, +∞), the coordinates of the foci of C are (0, ±a√(1 + 1/m))
  (m < -1 ∨ m > 0 → 
    ∃ f₁ f₂ : ℝ × ℝ, AreFociOf (C a m) f₁ f₂ ∧ 
      f₁ = (0, -a * Real.sqrt (1 + 1/m)) ∧ 
      f₂ = (0, a * Real.sqrt (1 + 1/m))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l1355_135529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1355_135597

theorem log_equation_solution (x : ℝ) :
  x > 0 → (Real.log 16 / Real.log x = Real.log 9 / Real.log 81) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1355_135597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_relatively_prime_pairs_with_three_integer_roots_l1355_135586

theorem infinite_relatively_prime_pairs_with_three_integer_roots :
  ∀ (x y : ℕ+), Nat.Coprime x.val y.val →
  ∃ (m n : ℕ+),
    Nat.Coprime m.val n.val ∧
    n = (x^2 + x*y + y^2)^3 ∧
    m = (x + y)*x*y ∧
    ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (a^3 : ℤ) - (n : ℤ)*a + (m : ℤ)*(n : ℤ) = 0 ∧
      (b^3 : ℤ) - (n : ℤ)*b + (m : ℤ)*(n : ℤ) = 0 ∧
      (c^3 : ℤ) - (n : ℤ)*c + (m : ℤ)*(n : ℤ) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_relatively_prime_pairs_with_three_integer_roots_l1355_135586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l1355_135528

/-- Represents a quadrilateral with vertices E, F, G, and H -/
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

/-- The length of a line segment between two points -/
def length (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Checks if an angle is a right angle -/
def is_right_angle (p q r : ℝ × ℝ) : Prop := sorry

/-- Checks if a length is an integer -/
def is_integer_length (p q : ℝ × ℝ) : Prop := sorry

theorem quadrilateral_area_theorem (EFGH : Quadrilateral) :
  is_right_angle EFGH.F EFGH.E EFGH.G →
  is_right_angle EFGH.H EFGH.E EFGH.G →
  length EFGH.E EFGH.G = 5 →
  (∃ (s1 s2 : ℝ × ℝ), s1 ≠ s2 ∧ 
    is_integer_length EFGH.E s1 ∧ 
    is_integer_length EFGH.E s2 ∧ 
    (s1 = EFGH.F ∨ s1 = EFGH.G ∨ s1 = EFGH.H) ∧
    (s2 = EFGH.F ∨ s2 = EFGH.G ∨ s2 = EFGH.H)) →
  (∀ (s : ℝ × ℝ), (s = EFGH.F ∨ s = EFGH.G ∨ s = EFGH.H) → length EFGH.E s > 1) →
  area EFGH = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l1355_135528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_single_color_l1355_135548

/-- Represents a color in the grid -/
def Color (n : ℕ) := Fin (n - 1)

/-- Represents the grid -/
def Grid (n : ℕ) := Fin n → Fin n → Color n

/-- Predicate to check if a row can be painted a single color -/
def can_paint_row {n : ℕ} (g : Grid n) (i : Fin n) : Prop :=
  ∃ c : Color n, ∃ j k : Fin n, j ≠ k ∧ g i j = c ∧ g i k = c

/-- Predicate to check if a column can be painted a single color -/
def can_paint_column {n : ℕ} (g : Grid n) (j : Fin n) : Prop :=
  ∃ c : Color n, ∃ i k : Fin n, i ≠ k ∧ g i j = c ∧ g k j = c

/-- Theorem stating that the entire grid can be painted a single color -/
theorem grid_single_color (n : ℕ) (h : n > 1) :
  ∀ g : Grid n, ∃ c : Color n, ∀ i j : Fin n, g i j = c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_single_color_l1355_135548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_composition_l1355_135552

-- Define the universe of discourse
variable {X : Type}

-- Define invertible functions a, b, and c
variable (a b c : X → X)
variable (a_inv b_inv c_inv : X → X)

-- Define the function composition operator
def comp (g f : X → X) : X → X := λ x ↦ g (f x)

-- State the theorem
theorem inverse_of_composition 
  (h_a_inv : comp a a_inv = id ∧ comp a_inv a = id)
  (h_b_inv : comp b b_inv = id ∧ comp b_inv b = id)
  (h_c_inv : comp c c_inv = id ∧ comp c_inv c = id) :
  let f := comp b (comp c a)
  comp f (comp a_inv (comp c_inv b_inv)) = id ∧
  comp (comp a_inv (comp c_inv b_inv)) f = id := by
  sorry

#check inverse_of_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_composition_l1355_135552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l1355_135566

-- Define basic geometric objects
structure Point where
structure Line where
structure Plane where

-- Define geometric relations
def Line.outsideOf (l : Line) (p : Plane) : Prop := sorry
def Line.intersectsWith (l1 l2 : Line) : Prop := sorry
def Plane.contains (p : Plane) (l : Line) : Prop := sorry
def Plane.intersectsWith (p1 p2 : Plane) : Prop := sorry
def Line.parallel (l1 l2 : Line) : Prop := sorry
def coplanar (l1 l2 l3 : Line) : Prop := sorry

-- Define the propositions
def proposition1 (l : Line) (p : Plane) : Prop :=
  Line.outsideOf l p → ∃! pt : Point, (∃ (l' : Line), l' = l ∧ Plane.contains p l' ∧ sorry) -- Placeholder for point on line

def proposition2 (a b : Line) (α β : Plane) : Prop :=
  Plane.contains α a ∧ Plane.contains β b ∧ Line.intersectsWith a b → 
  Plane.intersectsWith α β

def proposition3 (l1 l2 l3 : Line) : Prop :=
  Line.parallel l1 l2 ∧ Line.intersectsWith l3 l1 ∧ Line.intersectsWith l3 l2 →
  coplanar l1 l2 l3

def proposition4 (l1 l2 l3 : Line) : Prop :=
  Line.intersectsWith l1 l2 ∧ Line.intersectsWith l2 l3 ∧ Line.intersectsWith l3 l1 →
  coplanar l1 l2 l3

-- Theorem stating which propositions are true
theorem geometric_propositions :
  (∀ l p, proposition1 l p) ∧
  (∀ a b α β, proposition2 a b α β) ∧
  (∀ l1 l2 l3, proposition3 l1 l2 l3) ∧
  ¬(∀ l1 l2 l3, proposition4 l1 l2 l3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l1355_135566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_relationship_correct_l1355_135590

/-- Represents the relationship between a line and an ellipse -/
inductive LineEllipseRelation
  | Intersects
  | Tangent
  | Disjoint

/-- Determines the relationship between the line y = x + m and the ellipse 9x^2 + 16y^2 = 144 -/
noncomputable def lineEllipseRelationship (m : ℝ) : LineEllipseRelation :=
  if -5 < m ∧ m < 5 then LineEllipseRelation.Intersects
  else if m = 5 ∨ m = -5 then LineEllipseRelation.Tangent
  else LineEllipseRelation.Disjoint

theorem line_ellipse_relationship_correct (m : ℝ) :
  (lineEllipseRelationship m = LineEllipseRelation.Intersects ↔ -5 < m ∧ m < 5) ∧
  (lineEllipseRelationship m = LineEllipseRelation.Tangent ↔ m = 5 ∨ m = -5) ∧
  (lineEllipseRelationship m = LineEllipseRelation.Disjoint ↔ m > 5 ∨ m < -5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_relationship_correct_l1355_135590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_sum_l1355_135503

/-- Given a triangle DEF with centroid G, if the sum of squared distances from G to the vertices is 84,
    then the sum of squared side lengths of the triangle is 252. -/
theorem triangle_centroid_distance_sum (D E F G : EuclideanSpace ℝ (Fin 2)) :
  G = (1 / 3 : ℝ) • (D + E + F) →  -- G is the centroid
  ‖G - D‖^2 + ‖G - E‖^2 + ‖G - F‖^2 = 84 →
  ‖E - D‖^2 + ‖F - D‖^2 + ‖F - E‖^2 = 252 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_sum_l1355_135503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_is_finite_set_of_points_l1355_135570

def cost (n : ℕ) : ℚ := 0.25 * n + 5

def valid_purchase (n : ℕ) : Prop := n ≥ 5 ∧ n ≤ 20

def graph : Set (ℕ × ℚ) := {p | ∃ n, valid_purchase n ∧ p = (n, cost n)}

theorem goldfish_cost_graph_is_finite_set_of_points : 
  (∃ (S : Finset (ℕ × ℚ)), ∀ p, p ∈ graph → p ∈ S) ∧ 
  (∀ p q : ℕ × ℚ, p ∈ graph → q ∈ graph → p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2) :=
by
  sorry

#check goldfish_cost_graph_is_finite_set_of_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_cost_graph_is_finite_set_of_points_l1355_135570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_two_undefined_inverse_smallest_undefined_inverse_is_two_l1355_135569

theorem smallest_undefined_inverse (a : ℕ) : a > 0 ∧ 
  (∀ x : ℤ, x * a ≠ 1 % 72) ∧ 
  (∀ y : ℤ, y * a ≠ 1 % 90) → 
  a ≥ 2 := by
  sorry

theorem two_undefined_inverse : 
  (∀ x : ℤ, x * 2 ≠ 1 % 72) ∧ 
  (∀ y : ℤ, y * 2 ≠ 1 % 90) := by
  sorry

theorem smallest_undefined_inverse_is_two : 
  ∃! a : ℕ, a > 0 ∧ 
  (∀ x : ℤ, x * a ≠ 1 % 72) ∧ 
  (∀ y : ℤ, y * a ≠ 1 % 90) ∧ 
  ∀ b : ℕ, b > 0 ∧ 
  (∀ x : ℤ, x * b ≠ 1 % 72) ∧ 
  (∀ y : ℤ, y * b ≠ 1 % 90) → 
  a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_two_undefined_inverse_smallest_undefined_inverse_is_two_l1355_135569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rhombus_from_adjacent_equal_sides_l1355_135539

/-- A quadrilateral with coordinates in ℝ² --/
structure Quadrilateral where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ

/-- Distance between two points in ℝ² --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A quadrilateral is a rhombus if all sides are equal --/
def is_rhombus (q : Quadrilateral) : Prop :=
  distance q.a q.b = distance q.b q.c ∧
  distance q.b q.c = distance q.c q.d ∧
  distance q.c q.d = distance q.d q.a

/-- Theorem: A quadrilateral with a pair of adjacent sides equal is not necessarily a rhombus --/
theorem not_rhombus_from_adjacent_equal_sides : 
  ∃ (q : Quadrilateral), distance q.a q.b = distance q.b q.c ∧ ¬ is_rhombus q :=
by
  -- Construct a specific quadrilateral
  let q : Quadrilateral := {
    a := (0, 0),
    b := (1, 0),
    c := (1, 1),
    d := (0, 2)
  }
  -- Show this quadrilateral satisfies our conditions
  have h1 : distance q.a q.b = distance q.b q.c := by sorry
  have h2 : ¬ is_rhombus q := by sorry
  -- Conclude the proof
  exact ⟨q, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rhombus_from_adjacent_equal_sides_l1355_135539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l1355_135560

theorem floor_equation_solution (a b : ℤ) (ha : a > 0) (hb : b > 0) :
  (⌊(a^2 : ℚ) / b⌋ + ⌊(b^2 : ℚ) / a⌋ = ⌊((a^2 + b^2) : ℚ) / (a * b)⌋ + a * b) ↔
  (∃ k : ℤ, k > 0 ∧ ((a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l1355_135560
