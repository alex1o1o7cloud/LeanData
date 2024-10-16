import Mathlib

namespace NUMINAMATH_CALUDE_halves_to_one_and_half_l2428_242809

theorem halves_to_one_and_half :
  (3 : ℚ) / 2 / ((1 : ℚ) / 2) = 3 :=
sorry

end NUMINAMATH_CALUDE_halves_to_one_and_half_l2428_242809


namespace NUMINAMATH_CALUDE_forty_percent_of_sixty_minus_four_fifths_of_twenty_five_l2428_242892

theorem forty_percent_of_sixty_minus_four_fifths_of_twenty_five :
  (40 / 100 * 60) - (4 / 5 * 25) = 4 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_sixty_minus_four_fifths_of_twenty_five_l2428_242892


namespace NUMINAMATH_CALUDE_max_volume_triangular_cone_l2428_242838

/-- A quadrilateral cone with a square base -/
structure QuadrilateralCone where
  /-- Side length of the square base -/
  baseSideLength : ℝ
  /-- Sum of distances from apex to two adjacent vertices of the base -/
  sumOfDistances : ℝ

/-- Theorem: Maximum volume of triangular cone (A-BCM) -/
theorem max_volume_triangular_cone (cone : QuadrilateralCone) 
  (h1 : cone.baseSideLength = 6)
  (h2 : cone.sumOfDistances = 10) : 
  ∃ (v : ℝ), v = 24 ∧ ∀ (volume : ℝ), volume ≤ v :=
by
  sorry

end NUMINAMATH_CALUDE_max_volume_triangular_cone_l2428_242838


namespace NUMINAMATH_CALUDE_daycare_vegetable_preference_l2428_242832

theorem daycare_vegetable_preference (peas carrots corn : ℕ) 
  (h1 : peas = 6)
  (h2 : carrots = 9)
  (h3 : corn = 5) :
  (corn : ℚ) / (peas + carrots + corn : ℚ) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_daycare_vegetable_preference_l2428_242832


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2428_242851

theorem gcd_lcm_product (a b c : ℕ+) :
  let D := Nat.gcd a (Nat.gcd b c)
  let m := Nat.lcm a (Nat.lcm b c)
  D * m = a * b * c := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2428_242851


namespace NUMINAMATH_CALUDE_range_of_a_l2428_242898

theorem range_of_a (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ ∀ x ∈ s, (1 + a ≤ x ∧ x < 2)) → 
  (-5 < a ∧ a ≤ -4) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2428_242898


namespace NUMINAMATH_CALUDE_highest_average_speed_l2428_242823

def time_periods : Fin 5 → String
| 0 => "8-9 am"
| 1 => "9-10 am"
| 2 => "10-11 am"
| 3 => "2-3 pm"
| 4 => "3-4 pm"

def distances : Fin 5 → ℝ
| 0 => 50
| 1 => 70
| 2 => 60
| 3 => 80
| 4 => 40

def average_speed (i : Fin 5) : ℝ := distances i

def highest_speed_period : Fin 5 := 3

theorem highest_average_speed :
  ∀ (i : Fin 5), average_speed highest_speed_period ≥ average_speed i :=
by sorry

end NUMINAMATH_CALUDE_highest_average_speed_l2428_242823


namespace NUMINAMATH_CALUDE_problem_ratio_is_three_to_one_l2428_242816

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

/-- The number of problems Frank composes for each type -/
def frank_problems_per_type : ℕ := 30

/-- The total number of problems Frank composes -/
def frank_problems : ℕ := frank_problems_per_type * problem_types

/-- The ratio of problems Frank composes to problems Ryan composes -/
def problem_ratio : ℚ := frank_problems / ryan_problems

theorem problem_ratio_is_three_to_one : problem_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_ratio_is_three_to_one_l2428_242816


namespace NUMINAMATH_CALUDE_dove_hatching_fraction_l2428_242836

theorem dove_hatching_fraction (initial_doves : ℕ) (eggs_per_dove : ℕ) (total_doves_after : ℕ) :
  initial_doves = 20 →
  eggs_per_dove = 3 →
  total_doves_after = 65 →
  (total_doves_after - initial_doves : ℚ) / (initial_doves * eggs_per_dove) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_dove_hatching_fraction_l2428_242836


namespace NUMINAMATH_CALUDE_work_by_concurrent_forces_l2428_242866

/-- Work done by concurrent forces -/
theorem work_by_concurrent_forces :
  let F₁ : ℝ × ℝ := (Real.log 2, Real.log 2)
  let F₂ : ℝ × ℝ := (Real.log 5, Real.log 2)
  let s : ℝ × ℝ := (2 * Real.log 5, 1)
  let F : ℝ × ℝ := (F₁.1 + F₂.1, F₁.2 + F₂.2)
  let W : ℝ := F.1 * s.1 + F.2 * s.2
  W = 2 :=
by sorry

end NUMINAMATH_CALUDE_work_by_concurrent_forces_l2428_242866


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l2428_242802

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the domain
def domain : Set ℝ := {x : ℝ | x ≥ -1}

-- State the theorem
theorem f_geq_a_iff_a_in_range (a : ℝ) : 
  (∀ x ∈ domain, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l2428_242802


namespace NUMINAMATH_CALUDE_first_group_size_is_correct_l2428_242877

/-- The number of men in the first group -/
def first_group_size : ℕ := 20

/-- The length of the fountain built by the first group -/
def first_fountain_length : ℕ := 56

/-- The number of days taken by the first group to build their fountain -/
def first_group_days : ℕ := 14

/-- The number of men in the second group -/
def second_group_size : ℕ := 35

/-- The length of the fountain built by the second group -/
def second_fountain_length : ℕ := 21

/-- The number of days taken by the second group to build their fountain -/
def second_group_days : ℕ := 3

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_is_correct :
  (first_group_size : ℚ) * second_fountain_length * second_group_days =
  second_group_size * first_fountain_length * first_group_days :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_correct_l2428_242877


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l2428_242824

/-- Given a complex number z = 3 - i, prove that the imaginary part of z² is -6 -/
theorem imaginary_part_of_z_squared (z : ℂ) (h : z = 3 - I) : 
  (z^2).im = -6 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l2428_242824


namespace NUMINAMATH_CALUDE_range_of_a_l2428_242894

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + abs (x - a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a (1/2) ≥ f a x → x = 1/2) →
  (∀ x : ℝ, f a (-1/2) ≥ f a x → x = -1/2) →
  a > -1/2 ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2428_242894


namespace NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l2428_242853

theorem gold_copper_alloy_ratio (gold_density copper_density alloy_density : ℝ) 
  (hg : gold_density = 10)
  (hc : copper_density = 6)
  (ha : alloy_density = 8) :
  ∃ (g c : ℝ), g > 0 ∧ c > 0 ∧ 
    (gold_density * g + copper_density * c) / (g + c) = alloy_density ∧
    g = c := by
  sorry

end NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l2428_242853


namespace NUMINAMATH_CALUDE_kathryn_remaining_money_l2428_242852

/-- Calculates the remaining money after expenses for Kathryn -/
def remaining_money (initial_rent : ℕ) (salary : ℕ) : ℕ :=
  let food_travel : ℕ := 2 * initial_rent
  let new_rent : ℕ := initial_rent / 2
  let total_expenses : ℕ := new_rent + food_travel
  salary - total_expenses

/-- Proves that Kathryn's remaining money is $2000 -/
theorem kathryn_remaining_money :
  remaining_money 1200 5000 = 2000 := by
  sorry

#eval remaining_money 1200 5000

end NUMINAMATH_CALUDE_kathryn_remaining_money_l2428_242852


namespace NUMINAMATH_CALUDE_circle_regions_theorem_l2428_242847

/-- Represents the areas of regions in a circle circumscribed around a right triangle -/
structure CircleRegions where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sides of the right triangle -/
def triangle_sides : (ℝ × ℝ × ℝ) := (15, 20, 25)

/-- The circle is circumscribed around the triangle -/
axiom is_circumscribed (r : CircleRegions) : True

/-- C is the largest region (semicircle) -/
axiom C_is_largest (r : CircleRegions) : r.C ≥ r.A ∧ r.C ≥ r.B

/-- The area of the triangle -/
def triangle_area : ℝ := 150

/-- The theorem to prove -/
theorem circle_regions_theorem (r : CircleRegions) : 
  r.A + r.B + triangle_area = r.C :=
sorry

end NUMINAMATH_CALUDE_circle_regions_theorem_l2428_242847


namespace NUMINAMATH_CALUDE_sector_area_l2428_242819

theorem sector_area (r a b : ℝ) : 
  r = 1 →  -- radius is 1 cm
  a = 1 →  -- arc length is 1 cm
  b = (1/2) * r * a →  -- area formula for a sector
  b = 1/2  -- the area of the sector is 1/2 cm²
:= by sorry

end NUMINAMATH_CALUDE_sector_area_l2428_242819


namespace NUMINAMATH_CALUDE_john_kate_penny_difference_l2428_242880

theorem john_kate_penny_difference :
  ∀ (john_pennies kate_pennies : ℕ),
    john_pennies = 388 →
    kate_pennies = 223 →
    john_pennies - kate_pennies = 165 := by
  sorry

end NUMINAMATH_CALUDE_john_kate_penny_difference_l2428_242880


namespace NUMINAMATH_CALUDE_equation_is_ellipse_l2428_242882

def equation (x y : ℝ) : Prop :=
  x^2 + 2*y^2 - 6*x - 8*y + 9 = 0

def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ (x y : ℝ), f x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem equation_is_ellipse : is_ellipse equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_ellipse_l2428_242882


namespace NUMINAMATH_CALUDE_equation_solution_l2428_242856

theorem equation_solution (x y : ℕ) : x^y + y^x = 2408 ∧ x = 2407 → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2428_242856


namespace NUMINAMATH_CALUDE_sisters_and_brothers_in_family_l2428_242849

/-- Represents a family with boys and girls -/
structure Family where
  boys : Nat
  girls : Nat

/-- Calculates the number of sisters a girl has in the family (excluding herself) -/
def sisters_of_girl (f : Family) : Nat :=
  f.girls - 1

/-- Calculates the number of brothers a girl has in the family -/
def brothers_of_girl (f : Family) : Nat :=
  f.boys

theorem sisters_and_brothers_in_family (harry_sisters : Nat) (harry_brothers : Nat) :
  harry_sisters = 4 → harry_brothers = 3 →
  ∃ (f : Family),
    f.girls = harry_sisters + 1 ∧
    f.boys = harry_brothers + 1 ∧
    sisters_of_girl f = 3 ∧
    brothers_of_girl f = 3 :=
by sorry

end NUMINAMATH_CALUDE_sisters_and_brothers_in_family_l2428_242849


namespace NUMINAMATH_CALUDE_carSalesProfit_l2428_242844

/-- Represents the sale of a car with its selling price and profit/loss percentage -/
structure CarSale where
  sellingPrice : ℝ
  profitPercentage : ℝ

/-- Calculates the overall profit percentage for a list of car sales -/
def overallProfitPercentage (sales : List CarSale) : ℝ := sorry

/-- The main theorem stating the overall profit percentage for the given car sales -/
theorem carSalesProfit :
  let sales := [
    CarSale.mk 404415 15,
    CarSale.mk 404415 (-15),
    CarSale.mk 550000 10
  ]
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |overallProfitPercentage sales - 2.36| < ε := by sorry

end NUMINAMATH_CALUDE_carSalesProfit_l2428_242844


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_six_l2428_242850

/-- Triangle ABC with given properties -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The law of sines for a triangle -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- Theorem: In triangle ABC, if angle A = 120°, side a = 2, and side b = (2√3)/3, then angle B = π/6 -/
theorem angle_B_is_pi_over_six (t : Triangle) 
  (h1 : t.A = 2 * π / 3)  -- 120° in radians
  (h2 : t.a = 2)
  (h3 : t.b = 2 * Real.sqrt 3 / 3) :
  t.B = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_angle_B_is_pi_over_six_l2428_242850


namespace NUMINAMATH_CALUDE_inequality_proofs_l2428_242889

theorem inequality_proofs 
  (h : ∀ x > 0, 1 / (1 + x) < Real.log (1 + 1 / x) ∧ Real.log (1 + 1 / x) < 1 / x) :
  (1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 > Real.log 8) ∧
  (1/2 + 1/3 + 1/4 + 1/5 + 1/6 + 1/7 + 1/8 < Real.log 8) ∧
  ((1 : ℝ) / 1 + 8 / 8 + 28 / 64 + 56 / 512 + 70 / 4096 + 56 / 32768 + 28 / 262144 + 8 / 2097152 + 1 / 16777216 < Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l2428_242889


namespace NUMINAMATH_CALUDE_antonio_meatballs_l2428_242883

/-- Given a recipe for meatballs and family size, calculate how many meatballs Antonio will eat -/
theorem antonio_meatballs (hamburger_per_meatball : ℚ) (family_size : ℕ) (total_hamburger : ℕ) :
  hamburger_per_meatball = 1/8 →
  family_size = 8 →
  total_hamburger = 4 →
  (total_hamburger / hamburger_per_meatball) / family_size = 4 :=
by sorry

end NUMINAMATH_CALUDE_antonio_meatballs_l2428_242883


namespace NUMINAMATH_CALUDE_range_of_k_l2428_242845

def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  (∀ (x y : ℝ), x^2 / (4 - k) + y^2 / (1 - k) = 1 ↔ 
    x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (4 - k > 0) ∧ (1 - k < 0)

theorem range_of_k (k : ℝ) : 
  ((p k ∨ q k) ∧ ¬(p k ∧ q k)) → 
  ((-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l2428_242845


namespace NUMINAMATH_CALUDE_book_count_theorem_l2428_242833

/-- Represents the book collection of a person -/
structure BookCollection where
  initial : ℕ
  bought : ℕ
  lost : ℕ
  borrowed : ℕ

/-- Calculates the current number of books in a collection -/
def current_books (collection : BookCollection) : ℕ :=
  collection.initial - collection.lost

/-- Calculates the future number of books in a collection -/
def future_books (collection : BookCollection) : ℕ :=
  current_books collection + collection.bought + collection.borrowed

/-- Jason's book collection -/
def jason : BookCollection :=
  { initial := 18, bought := 8, lost := 0, borrowed := 0 }

/-- Mary's book collection -/
def mary : BookCollection :=
  { initial := 42, bought := 0, lost := 6, borrowed := 5 }

theorem book_count_theorem :
  (current_books jason + current_books mary = 54) ∧
  (future_books jason + future_books mary = 67) := by
  sorry

end NUMINAMATH_CALUDE_book_count_theorem_l2428_242833


namespace NUMINAMATH_CALUDE_inequality_proof_l2428_242828

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^3 + b^3) / 2 ≥ ((a^2 + b^2) / 2) * ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2428_242828


namespace NUMINAMATH_CALUDE_prob_two_consecutive_wins_l2428_242895

/-- The probability of player A winning exactly two consecutive games in a three-game series -/
theorem prob_two_consecutive_wins (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/4) (h2 : p2 = 1/3) (h3 : p3 = 1/3) : 
  p1 * p2 * (1 - p3) + (1 - p1) * p2 * p3 = 5/36 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_consecutive_wins_l2428_242895


namespace NUMINAMATH_CALUDE_max_value_theorem_l2428_242807

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 = 4) :
  ∃ (max : ℝ), max = (1 + Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z = x*y / (x + 2*y - 2) → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2428_242807


namespace NUMINAMATH_CALUDE_line_integral_equals_five_halves_l2428_242879

/-- Line segment from (0,0) to (4,3) -/
def L : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (4*t, 3*t)}

/-- The function to be integrated -/
def f (p : ℝ × ℝ) : ℝ := p.1 - p.2

theorem line_integral_equals_five_halves :
  ∫ p in L, f p = 5/2 := by sorry

end NUMINAMATH_CALUDE_line_integral_equals_five_halves_l2428_242879


namespace NUMINAMATH_CALUDE_octahedron_intersection_area_l2428_242876

/-- Represents a regular octahedron -/
structure RegularOctahedron where
  side_length : ℝ

/-- Represents the hexagonal intersection formed by a plane cutting the octahedron -/
structure HexagonalIntersection where
  octahedron : RegularOctahedron

/-- The area of the hexagonal intersection -/
def intersection_area (h : HexagonalIntersection) : ℝ := sorry

theorem octahedron_intersection_area 
  (o : RegularOctahedron)
  (h : HexagonalIntersection)
  (h_octahedron : h.octahedron = o)
  (side_length_eq : o.side_length = 2) :
  intersection_area h = 9 * Real.sqrt 3 / 8 := by sorry

end NUMINAMATH_CALUDE_octahedron_intersection_area_l2428_242876


namespace NUMINAMATH_CALUDE_min_value_of_f_l2428_242820

def f (x : ℝ) := 3 * x^2 - 6 * x + 9

theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2428_242820


namespace NUMINAMATH_CALUDE_paul_picked_72_cans_l2428_242817

/-- The number of cans Paul picked up on Saturday and Sunday --/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Paul picked up 72 cans in total --/
theorem paul_picked_72_cans :
  total_cans 6 3 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_paul_picked_72_cans_l2428_242817


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2428_242827

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2428_242827


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l2428_242805

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l2428_242805


namespace NUMINAMATH_CALUDE_factorial_difference_l2428_242890

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2428_242890


namespace NUMINAMATH_CALUDE_carmen_pets_difference_l2428_242846

/-- Given Carmen's initial number of cats and dogs, and the number of cats given up for adoption,
    prove that she now has 7 more cats than dogs. -/
theorem carmen_pets_difference (initial_cats initial_dogs cats_adopted : ℕ) 
    (h1 : initial_cats = 28)
    (h2 : initial_dogs = 18)
    (h3 : cats_adopted = 3) : 
  initial_cats - cats_adopted - initial_dogs = 7 := by
  sorry

end NUMINAMATH_CALUDE_carmen_pets_difference_l2428_242846


namespace NUMINAMATH_CALUDE_product_of_roots_l2428_242878

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 26 → ∃ y : ℝ, (x + 3) * (x - 4) = 26 ∧ (y + 3) * (y - 4) = 26 ∧ x * y = -38 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2428_242878


namespace NUMINAMATH_CALUDE_point_on_line_l2428_242860

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let A : Point := ⟨1, -5⟩
  let B : Point := ⟨3, -1⟩
  let C : Point := ⟨4.5, 2⟩
  collinear A B C := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2428_242860


namespace NUMINAMATH_CALUDE_school_meeting_attendance_l2428_242872

theorem school_meeting_attendance
  (seated_students : ℕ)
  (seated_teachers : ℕ)
  (standing_students : ℕ)
  (h1 : seated_students = 300)
  (h2 : seated_teachers = 30)
  (h3 : standing_students = 25) :
  seated_students + seated_teachers + standing_students = 355 :=
by sorry

end NUMINAMATH_CALUDE_school_meeting_attendance_l2428_242872


namespace NUMINAMATH_CALUDE_product_mod_800_l2428_242810

theorem product_mod_800 : (2437 * 2987) % 800 = 109 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_800_l2428_242810


namespace NUMINAMATH_CALUDE_cubic_roots_from_known_root_l2428_242855

/-- Given a cubic polynomial P(x) = x^3 + ax^2 + bx + c and a known root α,
    the other roots of P(x) are the roots of the quadratic polynomial Q(x)
    obtained by dividing P(x) by (x - α). -/
theorem cubic_roots_from_known_root (a b c α : ℝ) :
  (α^3 + a*α^2 + b*α + c = 0) →
  ∃ (p q : ℝ),
    (∀ x, x^3 + a*x^2 + b*x + c = (x - α) * (x^2 + p*x + q)) ∧
    (∀ x, x ≠ α ∧ x^3 + a*x^2 + b*x + c = 0 ↔ x^2 + p*x + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_from_known_root_l2428_242855


namespace NUMINAMATH_CALUDE_equidistant_line_theorem_l2428_242811

-- Define the points
def P : ℝ × ℝ := (1, 2)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -5)

-- Define the property of the line being equidistant from A and B
def is_equidistant (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), l x y → (abs (3 * x + 2 * y - 7) = abs (4 * x + y - 6))

-- Define the two possible line equations
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 7 = 0
def line2 (x y : ℝ) : Prop := 4 * x + y - 6 = 0

-- Theorem statement
theorem equidistant_line_theorem :
  ∃ (l : ℝ → ℝ → Prop), 
    (l P.1 P.2) ∧ 
    (is_equidistant l) ∧ 
    (∀ (x y : ℝ), l x y ↔ (line1 x y ∨ line2 x y)) :=
sorry

end NUMINAMATH_CALUDE_equidistant_line_theorem_l2428_242811


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2428_242842

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0) (pos_q : q > 0) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq : ∃ d : ℝ, b = p + d ∧ c = p + 2*d ∧ q = p + 3*d)
  : ∀ x : ℝ, b * x^2 - 2*a * x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2428_242842


namespace NUMINAMATH_CALUDE_salon_cost_calculation_l2428_242891

def salon_total_cost (manicure_cost pedicure_cost hair_treatment_cost : ℝ)
                     (manicure_tax_rate pedicure_tax_rate hair_treatment_tax_rate : ℝ)
                     (manicure_tip_rate pedicure_tip_rate hair_treatment_tip_rate : ℝ) : ℝ :=
  let manicure_total := manicure_cost * (1 + manicure_tax_rate + manicure_tip_rate)
  let pedicure_total := pedicure_cost * (1 + pedicure_tax_rate + pedicure_tip_rate)
  let hair_treatment_total := hair_treatment_cost * (1 + hair_treatment_tax_rate + hair_treatment_tip_rate)
  manicure_total + pedicure_total + hair_treatment_total

theorem salon_cost_calculation :
  salon_total_cost 30 40 50 0.05 0.07 0.09 0.25 0.20 0.15 = 151.80 := by
  sorry

end NUMINAMATH_CALUDE_salon_cost_calculation_l2428_242891


namespace NUMINAMATH_CALUDE_jolene_babysitting_charge_l2428_242886

theorem jolene_babysitting_charge 
  (num_families : ℕ) 
  (num_cars : ℕ) 
  (car_wash_fee : ℚ) 
  (total_raised : ℚ) :
  num_families = 4 →
  num_cars = 5 →
  car_wash_fee = 12 →
  total_raised = 180 →
  (num_families : ℚ) * (total_raised - num_cars * car_wash_fee) / num_families = 30 := by
  sorry

end NUMINAMATH_CALUDE_jolene_babysitting_charge_l2428_242886


namespace NUMINAMATH_CALUDE_f_negative_when_x_greater_than_one_third_l2428_242821

def f (x : ℝ) := -3 * x + 1

theorem f_negative_when_x_greater_than_one_third :
  ∀ x : ℝ, x > 1/3 → f x < 0 := by
sorry

end NUMINAMATH_CALUDE_f_negative_when_x_greater_than_one_third_l2428_242821


namespace NUMINAMATH_CALUDE_system_solutions_l2428_242843

def is_solution (x y : ℤ) : Prop :=
  |x^2 - 2*x| < y + 1/2 ∧ y + |x - 1| < 2

theorem system_solutions :
  {(x, y) : ℤ × ℤ | is_solution x y} = {(0, 0), (2, 0), (1, 1)} :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2428_242843


namespace NUMINAMATH_CALUDE_mrs_hilt_remaining_cents_l2428_242888

/-- Given that Mrs. Hilt had 15 cents initially and spent 11 cents on a pencil, 
    prove that she was left with 4 cents. -/
theorem mrs_hilt_remaining_cents 
  (initial_cents : ℕ) 
  (pencil_cost : ℕ) 
  (h1 : initial_cents = 15)
  (h2 : pencil_cost = 11) :
  initial_cents - pencil_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_remaining_cents_l2428_242888


namespace NUMINAMATH_CALUDE_one_positive_real_solution_l2428_242874

def f (x : ℝ) := x^4 + 8*x^3 + 16*x^2 + 2023*x - 2023

theorem one_positive_real_solution :
  ∃! x : ℝ, x > 0 ∧ x^10 + 8*x^9 + 16*x^8 + 2023*x^7 - 2023*x^6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_one_positive_real_solution_l2428_242874


namespace NUMINAMATH_CALUDE_sum_of_angles_x_and_y_l2428_242825

-- Define a circle divided into 16 equal arcs
def circle_arcs : ℕ := 16

-- Define the span of angle x
def x_span : ℕ := 3

-- Define the span of angle y
def y_span : ℕ := 5

-- Theorem statement
theorem sum_of_angles_x_and_y (x y : Real) :
  (x = (360 / circle_arcs * x_span) / 2) →
  (y = (360 / circle_arcs * y_span) / 2) →
  x + y = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_x_and_y_l2428_242825


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2428_242840

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 0, -4; 3, -1, 5; 1, 2, 3]
  Matrix.det A = -54 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2428_242840


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2428_242854

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 7 * x + 3
  let solution_set : Set ℝ := {x | f x > 0}
  solution_set = {x | x < -3 ∨ x > -0.5} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2428_242854


namespace NUMINAMATH_CALUDE_quadratic_intersection_at_one_point_l2428_242884

theorem quadratic_intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 5 * x + 3 = -2 * x - 2) ↔ b = 49 / 20 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_at_one_point_l2428_242884


namespace NUMINAMATH_CALUDE_fraction_sum_l2428_242812

theorem fraction_sum (a b : ℚ) (h : a / b = 2 / 5) : (a + b) / b = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2428_242812


namespace NUMINAMATH_CALUDE_smallest_bases_sum_is_correct_l2428_242897

/-- Represents a number in a given base -/
def representationInBase (n : ℕ) (base : ℕ) : ℕ := 
  (n / base) * base + (n % base)

/-- The smallest possible sum of bases c and d where 83 in base c equals 38 in base d -/
def smallestBasesSum : ℕ := 27

theorem smallest_bases_sum_is_correct :
  ∀ c d : ℕ, c ≥ 2 → d ≥ 2 →
  representationInBase 83 c = representationInBase 38 d →
  c + d ≥ smallestBasesSum :=
sorry

end NUMINAMATH_CALUDE_smallest_bases_sum_is_correct_l2428_242897


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_sequence_l2428_242887

/-- Given a geometric sequence with first term a, common ratio r, and n-th term defined as a * r^(n-1) -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- The fifth term of a geometric sequence with first term 25 and common ratio -2 is 400 -/
theorem fifth_term_of_specific_geometric_sequence :
  let a := 25
  let r := -2
  geometric_sequence a r 5 = 400 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_sequence_l2428_242887


namespace NUMINAMATH_CALUDE_modular_congruence_existence_l2428_242873

theorem modular_congruence_existence (a c : ℕ+) (b : ℤ) :
  ∃ x : ℕ+, (c : ℤ) ∣ ((a : ℤ)^(x : ℕ) + x - b) := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_existence_l2428_242873


namespace NUMINAMATH_CALUDE_first_box_not_empty_count_l2428_242800

/-- The number of ways to distribute three distinct balls into four boxes. -/
def total_distributions : ℕ := 4^3

/-- The number of ways to distribute three distinct balls into four boxes
    such that the first box is empty. -/
def distributions_with_empty_first_box : ℕ := 3^3

theorem first_box_not_empty_count :
  total_distributions - distributions_with_empty_first_box = 37 := by
  sorry

end NUMINAMATH_CALUDE_first_box_not_empty_count_l2428_242800


namespace NUMINAMATH_CALUDE_inequality_solution_minimum_value_and_points_l2428_242881

-- Part 1
def solution_set := {x : ℝ | 0 < x ∧ x < 2}

theorem inequality_solution : 
  ∀ x : ℝ, |2*x - 1| < |x| + 1 ↔ x ∈ solution_set :=
sorry

-- Part 2
def constraint (x y z : ℝ) := x^2 + y^2 + z^2 = 4

theorem minimum_value_and_points :
  ∃ (x y z : ℝ), constraint x y z ∧
    (∀ (a b c : ℝ), constraint a b c → x - 2*y + 2*z ≤ a - 2*b + 2*c) ∧
    x - 2*y + 2*z = -6 ∧
    x = -2/3 ∧ y = 4/3 ∧ z = -4/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_minimum_value_and_points_l2428_242881


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2428_242871

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k - 3) * x^2 - 4 * x + 2 = 0) ↔ k ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2428_242871


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l2428_242863

/-- Represents a keystone arch configuration -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_are_congruent : Bool
  trapezoids_are_isosceles : Bool
  sides_meet_at_center : Bool

/-- Calculates the larger interior angle of a trapezoid in the keystone arch -/
def larger_interior_angle (arch : KeystoneArch) : ℝ :=
  sorry

/-- Theorem stating that the larger interior angle of each trapezoid in a 12-piece keystone arch is 97.5° -/
theorem keystone_arch_angle (arch : KeystoneArch) :
  arch.num_trapezoids = 12 ∧ 
  arch.trapezoids_are_congruent ∧ 
  arch.trapezoids_are_isosceles ∧ 
  arch.sides_meet_at_center →
  larger_interior_angle arch = 97.5 :=
sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l2428_242863


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2428_242893

/-- Given a trip with specified distances and speeds, calculate the average speed -/
theorem average_speed_calculation (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 350)
  (h2 : distance1 = 200)
  (h3 : speed1 = 20)
  (h4 : distance2 = total_distance - distance1)
  (h5 : speed2 = 15) :
  (total_distance) / ((distance1 / speed1) + (distance2 / speed2)) = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2428_242893


namespace NUMINAMATH_CALUDE_root_expression_equals_five_l2428_242867

theorem root_expression_equals_five (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h1 : a - 5 * Real.sqrt a + 2 = 0)
  (h2 : b - 5 * Real.sqrt b + 2 = 0) :
  (a * Real.sqrt a + b * Real.sqrt b) / (a - b) *
  (2 / Real.sqrt a - 2 / Real.sqrt b) /
  (Real.sqrt a - (a + b) / Real.sqrt b) +
  5 * (5 * Real.sqrt a - a) / (b + 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_root_expression_equals_five_l2428_242867


namespace NUMINAMATH_CALUDE_meal_serving_problem_l2428_242834

/-- The number of derangements of n elements -/
def subfactorial (n : ℕ) : ℕ := sorry

/-- The number of ways to serve meals to exactly three people correctly -/
def waysToServeThreeCorrectly (totalPeople : ℕ) (mealTypes : ℕ) (peoplePerMeal : ℕ) : ℕ :=
  Nat.choose totalPeople 3 * subfactorial (totalPeople - 3)

theorem meal_serving_problem :
  waysToServeThreeCorrectly 15 3 5 = 80157776755 := by sorry

end NUMINAMATH_CALUDE_meal_serving_problem_l2428_242834


namespace NUMINAMATH_CALUDE_length_comparison_l2428_242862

theorem length_comparison : 
  900/1000 < (2 : ℝ) ∧ (2 : ℝ) < 300/100 ∧ 300/100 < 80/10 ∧ 80/10 < 1000 := by
  sorry

end NUMINAMATH_CALUDE_length_comparison_l2428_242862


namespace NUMINAMATH_CALUDE_book_purchase_equations_l2428_242841

/-- Represents the problem of students pooling money to buy a book. -/
theorem book_purchase_equations (x y : ℝ) :
  (∀ (excess shortage : ℝ),
    excess = 4 ∧ shortage = 3 →
    (9 * x - y = excess ∧ y - 8 * x = shortage)) ↔
  (9 * x - y = 4 ∧ y - 8 * x = 3) :=
sorry

end NUMINAMATH_CALUDE_book_purchase_equations_l2428_242841


namespace NUMINAMATH_CALUDE_rug_coverage_area_l2428_242839

/-- Given three rugs with specified overlapping areas, calculate the total floor area covered. -/
theorem rug_coverage_area (total_rug_area double_layer triple_layer : ℝ) 
  (h1 : total_rug_area = 212)
  (h2 : double_layer = 24)
  (h3 : triple_layer = 24) :
  total_rug_area - double_layer - 2 * triple_layer = 140 :=
by sorry

end NUMINAMATH_CALUDE_rug_coverage_area_l2428_242839


namespace NUMINAMATH_CALUDE_health_run_distance_to_finish_l2428_242875

/-- The distance between a runner and the finish line in a health run event -/
def distance_to_finish (total_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  total_distance - speed * time

/-- Theorem: In a 7.5 km health run, after running for 10 minutes at speed x km/min, 
    the distance to the finish line is 7.5 - 10x km -/
theorem health_run_distance_to_finish (x : ℝ) : 
  distance_to_finish 7.5 x 10 = 7.5 - 10 * x := by
  sorry

end NUMINAMATH_CALUDE_health_run_distance_to_finish_l2428_242875


namespace NUMINAMATH_CALUDE_expected_sides_formula_rectangle_limit_sides_l2428_242857

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents the state after cutting a polygon -/
structure CutState where
  initialPolygon : Polygon
  numCuts : ℕ

/-- Calculates the expected number of sides after cuts -/
def expectedSides (state : CutState) : ℚ :=
  (state.initialPolygon.sides + 4 * state.numCuts) / (state.numCuts + 1)

/-- Theorem: The expected number of sides after cuts is (n + 4k) / (k + 1) -/
theorem expected_sides_formula (state : CutState) :
  expectedSides state = (state.initialPolygon.sides + 4 * state.numCuts) / (state.numCuts + 1) := by
  sorry

/-- Corollary: For a rectangle (4 sides), as cuts approach infinity, expected sides approach 4 -/
theorem rectangle_limit_sides (initialRect : Polygon) (h : initialRect.sides = 4) :
  ∀ ε > 0, ∃ N, ∀ k ≥ N,
    |expectedSides { initialPolygon := initialRect, numCuts := k } - 4| < ε := by
  sorry

end NUMINAMATH_CALUDE_expected_sides_formula_rectangle_limit_sides_l2428_242857


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2428_242818

/-- An arithmetic sequence with a non-zero common difference and positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a n > 0
  h3 : ∀ n, a (n + 1) = a n + d

/-- For an arithmetic sequence with non-zero common difference and positive terms, a₁ · a₈ < a₄ · a₅ -/
theorem arithmetic_sequence_inequality (seq : ArithmeticSequence) : seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2428_242818


namespace NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l2428_242861

-- Define the function g
def g (c : ℤ) : ℝ → ℝ := λ x => 4 * x + c

-- State the theorem
theorem intersection_point_of_function_and_inverse (c : ℤ) :
  ∃ (d : ℤ), (g c (-4) = d ∧ g c d = -4) → d = -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l2428_242861


namespace NUMINAMATH_CALUDE_cos_cube_decomposition_l2428_242804

theorem cos_cube_decomposition (b₁ b₂ b₃ : ℝ) :
  (∀ θ : ℝ, Real.cos θ ^ 3 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ)) →
  b₁^2 + b₂^2 + b₃^2 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_cube_decomposition_l2428_242804


namespace NUMINAMATH_CALUDE_original_number_proof_l2428_242801

theorem original_number_proof (x : ℝ) : 
  (x + 0.375 * x) - (x - 0.425 * x) = 85 → x = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2428_242801


namespace NUMINAMATH_CALUDE_subset_intersection_condition_solution_set_eq_interval_l2428_242831

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- Define the theorem
theorem subset_intersection_condition (a : ℝ) :
  (A a).Nonempty ∧ (A a) ⊆ (A a) ∩ B ↔ 6 ≤ a ∧ a ≤ 9 := by
  sorry

-- Define the set of all 'a' that satisfies the condition
def solution_set : Set ℝ := {a | (A a).Nonempty ∧ (A a) ⊆ (A a) ∩ B}

-- Prove that the solution set is equal to the interval [6, 9]
theorem solution_set_eq_interval :
  solution_set = {a | 6 ≤ a ∧ a ≤ 9} := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_condition_solution_set_eq_interval_l2428_242831


namespace NUMINAMATH_CALUDE_binary_1100_is_12_l2428_242813

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + (if bit then 2^i else 0)) 0

theorem binary_1100_is_12 : 
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_is_12_l2428_242813


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2428_242815

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 3*I
  z₁ / z₂ = 9/13 - (19/13)*I := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2428_242815


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2428_242835

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion term
def expansionTerm (a : ℝ) (r : ℕ) : ℝ := 
  (-1)^r * a^(8 - r) * binomial 8 r

-- State the theorem
theorem expansion_coefficient (a : ℝ) : 
  (expansionTerm a 4 = 70) → (a = 1 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2428_242835


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2428_242814

theorem factorization_of_cubic (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2428_242814


namespace NUMINAMATH_CALUDE_inequality_proof_l2428_242837

theorem inequality_proof (n : ℕ) (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_max : a = max a (max b (max c (max x (max y z)))))
  (h_sum : a + b + c = x + y + z)
  (h_prod : a * b * c = x * y * z) :
  a^n + b^n + c^n ≥ x^n + y^n + z^n := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2428_242837


namespace NUMINAMATH_CALUDE_percentage_b_grades_l2428_242822

def scores : List Nat := [91, 82, 56, 99, 86, 95, 88, 79, 77, 68, 83, 81, 65, 84, 93, 72, 89, 78]

def is_b_grade (score : Nat) : Bool := 85 ≤ score ∧ score ≤ 93

def count_b_grades (scores : List Nat) : Nat :=
  scores.filter is_b_grade |>.length

theorem percentage_b_grades :
  let total_students := scores.length
  let b_grade_students := count_b_grades scores
  (b_grade_students : Rat) / total_students * 100 = 27.78 := by
  sorry

end NUMINAMATH_CALUDE_percentage_b_grades_l2428_242822


namespace NUMINAMATH_CALUDE_square_sum_of_linear_equations_l2428_242848

theorem square_sum_of_linear_equations (x y : ℝ) 
  (eq1 : 3 * x + y = 20) 
  (eq2 : 4 * x + y = 25) : 
  x^2 + y^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_linear_equations_l2428_242848


namespace NUMINAMATH_CALUDE_siblings_age_ratio_l2428_242869

theorem siblings_age_ratio : 
  ∀ (henry_age sister_age : ℕ),
  henry_age = 4 * sister_age →
  henry_age + sister_age + 15 = 240 →
  sister_age / 15 = 3 := by
sorry

end NUMINAMATH_CALUDE_siblings_age_ratio_l2428_242869


namespace NUMINAMATH_CALUDE_all_log_monotonic_exists_divisible_by_2_and_5_exists_log2_positive_all_statements_true_l2428_242865

-- Define logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- 1. All logarithmic functions are monotonic
theorem all_log_monotonic (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  StrictMono (log a) := by sorry

-- 2. There exists an integer divisible by both 2 and 5
theorem exists_divisible_by_2_and_5 :
  ∃ n : ℤ, 2 ∣ n ∧ 5 ∣ n := by sorry

-- 3. There exists a real number x such that log₂x > 0
theorem exists_log2_positive :
  ∃ x : ℝ, log 2 x > 0 := by sorry

-- All statements are true
theorem all_statements_true :
  (∀ a : ℝ, a > 0 → a ≠ 1 → StrictMono (log a)) ∧
  (∃ n : ℤ, 2 ∣ n ∧ 5 ∣ n) ∧
  (∃ x : ℝ, log 2 x > 0) := by sorry

end NUMINAMATH_CALUDE_all_log_monotonic_exists_divisible_by_2_and_5_exists_log2_positive_all_statements_true_l2428_242865


namespace NUMINAMATH_CALUDE_nonagon_diagonals_count_l2428_242808

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of vertices each vertex can connect to (excluding itself and adjacent vertices) -/
def connections_per_vertex : ℕ := nonagon_sides - 3

theorem nonagon_diagonals_count :
  num_diagonals_nonagon = (nonagon_sides * connections_per_vertex) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_count_l2428_242808


namespace NUMINAMATH_CALUDE_school_pet_ownership_l2428_242829

theorem school_pet_ownership (total_students : ℕ) (cat_owners : ℕ) (bird_owners : ℕ)
  (h_total : total_students = 500)
  (h_cats : cat_owners = 80)
  (h_birds : bird_owners = 120) :
  (cat_owners : ℚ) / total_students * 100 = 16 ∧
  (bird_owners : ℚ) / total_students * 100 = 24 :=
by sorry

end NUMINAMATH_CALUDE_school_pet_ownership_l2428_242829


namespace NUMINAMATH_CALUDE_lcm_of_40_and_14_l2428_242864

theorem lcm_of_40_and_14 :
  let n : ℕ := 40
  let m : ℕ := 14
  let gcf : ℕ := 10
  Nat.gcd n m = gcf →
  Nat.lcm n m = 56 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_40_and_14_l2428_242864


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l2428_242806

theorem five_digit_divisible_by_nine :
  ∀ B : ℕ,
  (0 ≤ B ∧ B ≤ 9) →
  (40000 + 10000*B + 1000*B + 100 + 10 + 3) % 9 = 0 →
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l2428_242806


namespace NUMINAMATH_CALUDE_parallelepiped_net_theorem_l2428_242868

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Represents a net of a parallelepiped -/
structure Net :=
  (squares : ℕ)

/-- Function to unfold a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- Function to remove one square from a net -/
def remove_square (n : Net) : Net :=
  { squares := n.squares - 1 }

/-- Theorem stating that a 2 × 1 × 1 parallelepiped unfolds into a net with 10 squares,
    and removing one square results in a valid net with 9 squares -/
theorem parallelepiped_net_theorem :
  let p : Parallelepiped := ⟨2, 1, 1⟩
  let full_net : Net := unfold p
  let cut_net : Net := remove_square full_net
  full_net.squares = 10 ∧ cut_net.squares = 9 := by
  sorry


end NUMINAMATH_CALUDE_parallelepiped_net_theorem_l2428_242868


namespace NUMINAMATH_CALUDE_goldfish_ratio_l2428_242830

/-- Proves the ratio of goldfish Bexley brought to Hershel's initial goldfish -/
theorem goldfish_ratio :
  ∀ (hershel_betta hershel_goldfish bexley_goldfish : ℕ),
  hershel_betta = 10 →
  hershel_goldfish = 15 →
  ∃ (total_after_gift : ℕ),
    total_after_gift = 17 ∧
    (hershel_betta + (2 / 5 : ℚ) * hershel_betta + hershel_goldfish + bexley_goldfish) / 2 = total_after_gift →
    bexley_goldfish * 3 = hershel_goldfish :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_ratio_l2428_242830


namespace NUMINAMATH_CALUDE_dance_workshop_avg_age_children_l2428_242858

theorem dance_workshop_avg_age_children (total_participants : ℕ) 
  (overall_avg_age : ℚ) (num_women : ℕ) (num_men : ℕ) (num_children : ℕ) 
  (avg_age_women : ℚ) (avg_age_men : ℚ) 
  (h1 : total_participants = 50)
  (h2 : overall_avg_age = 20)
  (h3 : num_women = 30)
  (h4 : num_men = 10)
  (h5 : num_children = 10)
  (h6 : avg_age_women = 22)
  (h7 : avg_age_men = 25)
  (h8 : total_participants = num_women + num_men + num_children) :
  (total_participants * overall_avg_age - num_women * avg_age_women - num_men * avg_age_men) / num_children = 9 := by
  sorry

end NUMINAMATH_CALUDE_dance_workshop_avg_age_children_l2428_242858


namespace NUMINAMATH_CALUDE_revenue_decrease_l2428_242896

def previous_revenue : ℝ := 69.0
def decrease_percentage : ℝ := 30.434782608695656

theorem revenue_decrease (previous_revenue : ℝ) (decrease_percentage : ℝ) :
  previous_revenue * (1 - decrease_percentage / 100) = 48.0 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l2428_242896


namespace NUMINAMATH_CALUDE_officer_average_salary_l2428_242803

theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h1 : total_avg = 120)
  (h2 : non_officer_avg = 110)
  (h3 : officer_count = 15)
  (h4 : non_officer_count = 525) :
  let total_count := officer_count + non_officer_count
  let officer_total := total_avg * total_count - non_officer_avg * non_officer_count
  officer_total / officer_count = 470 := by
sorry

end NUMINAMATH_CALUDE_officer_average_salary_l2428_242803


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2428_242826

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 5*x > 22) → x ≤ -4 ∧ (7 - 5*(-4) > 22) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2428_242826


namespace NUMINAMATH_CALUDE_volume_of_extended_box_l2428_242899

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def extendedVolume (b : Box) : ℝ :=
  sorry

/-- The specific box in the problem -/
def problemBox : Box :=
  { length := 4, width := 5, height := 6 }

/-- The theorem to be proved -/
theorem volume_of_extended_box :
  extendedVolume problemBox = (804 + 139 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_extended_box_l2428_242899


namespace NUMINAMATH_CALUDE_dihedral_angle_bounds_l2428_242885

/-- A regular pyramid with an n-sided polygonal base -/
structure RegularPyramid where
  n : ℕ
  base_sides : n > 2

/-- The dihedral angle between two adjacent lateral faces of a regular pyramid -/
def dihedral_angle (p : RegularPyramid) : ℝ :=
  sorry

/-- Theorem: The dihedral angle in a regular pyramid is bounded -/
theorem dihedral_angle_bounds (p : RegularPyramid) :
  (((p.n - 2) / p.n : ℝ) * Real.pi) < dihedral_angle p ∧ dihedral_angle p < Real.pi :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_bounds_l2428_242885


namespace NUMINAMATH_CALUDE_unique_prime_with_square_divisor_sum_l2428_242859

theorem unique_prime_with_square_divisor_sum : 
  ∃! p : ℕ, Prime p ∧ 
  ∃ n : ℕ, (1 + p + p^2 + p^3 + p^4 : ℕ) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_square_divisor_sum_l2428_242859


namespace NUMINAMATH_CALUDE_square_sum_from_means_l2428_242870

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20)
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 96) :
  a^2 + b^2 = 1408 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l2428_242870
