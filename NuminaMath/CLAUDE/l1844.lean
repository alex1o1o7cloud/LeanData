import Mathlib

namespace NUMINAMATH_CALUDE_log_equation_solution_l1844_184436

/-- Given a > 0, prove that the solution to log_√2(x - a) = 1 + log_2 x is x = a + 1 + √(2a + 1) -/
theorem log_equation_solution (a : ℝ) (ha : a > 0) :
  ∃! x : ℝ, x > a ∧ Real.log (x - a) / Real.log (Real.sqrt 2) = 1 + Real.log x / Real.log 2 ∧
  x = a + 1 + Real.sqrt (2 * a + 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1844_184436


namespace NUMINAMATH_CALUDE_triangle_not_tileable_with_sphinx_l1844_184467

/-- Represents a triangle that can be divided into smaller triangles -/
structure DivisibleTriangle where
  side_length : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- Represents a sphinx tile -/
structure SphinxTile where
  covers_even_orientations : Bool

/-- Checks if a triangle can be tiled with sphinx tiles -/
def can_be_tiled_with_sphinx (t : DivisibleTriangle) (s : SphinxTile) : Prop :=
  s.covers_even_orientations →
    (t.upward_triangles % 2 = 0 ∧ t.downward_triangles % 2 = 0)

/-- The main theorem stating that the specific triangle cannot be tiled with sphinx tiles -/
theorem triangle_not_tileable_with_sphinx :
  ∀ (t : DivisibleTriangle) (s : SphinxTile),
    t.side_length = 6 →
    t.upward_triangles = 21 →
    t.downward_triangles = 15 →
    s.covers_even_orientations →
    ¬(can_be_tiled_with_sphinx t s) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_not_tileable_with_sphinx_l1844_184467


namespace NUMINAMATH_CALUDE_triangle_angle_C_l1844_184478

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if A = π/6, a = 1, and b = √3, then C = π/2 -/
theorem triangle_angle_C (A B C a b c : Real) : 
  A = π/6 → a = 1 → b = Real.sqrt 3 → 
  a / Real.sin A = b / Real.sin B →
  A + B + C = π →
  C = π/2 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l1844_184478


namespace NUMINAMATH_CALUDE_new_years_eve_appetizer_cost_l1844_184404

def cost_per_person (chips_cost creme_fraiche_cost caviar_cost : ℚ) (num_people : ℕ) : ℚ :=
  (chips_cost + creme_fraiche_cost + caviar_cost) / num_people

theorem new_years_eve_appetizer_cost :
  cost_per_person 3 5 73 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_new_years_eve_appetizer_cost_l1844_184404


namespace NUMINAMATH_CALUDE_triangle_parallel_lines_l1844_184417

theorem triangle_parallel_lines (base : ℝ) (h1 : base = 20) : 
  ∀ (line1 line2 : ℝ),
    (line1 / base)^2 = 1/4 →
    (line2 / line1)^2 = 1/3 →
    line2 = 10 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_parallel_lines_l1844_184417


namespace NUMINAMATH_CALUDE_tunnel_length_l1844_184431

/-- The length of a tunnel given train passage information -/
theorem tunnel_length (train_length : ℝ) (total_time : ℝ) (inside_time : ℝ) : 
  train_length = 300 →
  total_time = 60 →
  inside_time = 30 →
  ∃ (tunnel_length : ℝ) (train_speed : ℝ),
    tunnel_length + train_length = total_time * train_speed ∧
    tunnel_length - train_length = inside_time * train_speed ∧
    tunnel_length = 900 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l1844_184431


namespace NUMINAMATH_CALUDE_cubic_fraction_inequality_l1844_184423

theorem cubic_fraction_inequality (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1/2 ≤ (a^3 + b^3) / (a^2 + b^2) ∧ (a^3 + b^3) / (a^2 + b^2) ≤ 1 ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1/2 ↔ a = 1/2 ∧ b = 1/2) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0)) :=
sorry

end NUMINAMATH_CALUDE_cubic_fraction_inequality_l1844_184423


namespace NUMINAMATH_CALUDE_closest_to_one_l1844_184497

theorem closest_to_one : 
  let numbers : List ℝ := [3/4, 1.2, 0.81, 4/3, 7/10]
  ∀ x ∈ numbers, |0.81 - 1| ≤ |x - 1| := by
sorry

end NUMINAMATH_CALUDE_closest_to_one_l1844_184497


namespace NUMINAMATH_CALUDE_cube_order_l1844_184457

theorem cube_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_l1844_184457


namespace NUMINAMATH_CALUDE_total_shared_amount_l1844_184416

/-- Represents the money sharing problem with three people --/
structure MoneySharing where
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ
  share1 : ℕ

/-- Theorem stating that given the conditions, the total shared amount is 195 --/
theorem total_shared_amount (ms : MoneySharing) 
  (h1 : ms.ratio1 = 2)
  (h2 : ms.ratio2 = 3)
  (h3 : ms.ratio3 = 8)
  (h4 : ms.share1 = 30) :
  ms.share1 + (ms.share1 / ms.ratio1 * ms.ratio2) + (ms.share1 / ms.ratio1 * ms.ratio3) = 195 := by
  sorry

#check total_shared_amount

end NUMINAMATH_CALUDE_total_shared_amount_l1844_184416


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1844_184408

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (min_val : Real), min_val = (11 * Real.sqrt 2) / 2 ∧
  ∀ θ', 0 < θ' ∧ θ' < π/2 →
    3 * cos θ' + 2 / sin θ' + 2 * Real.sqrt 2 * tan θ' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1844_184408


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l1844_184490

theorem fifteenth_student_age 
  (total_students : ℕ) 
  (avg_age_all : ℝ) 
  (num_group1 : ℕ) 
  (avg_age_group1 : ℝ) 
  (num_group2 : ℕ) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 5)
  (h4 : avg_age_group1 = 13)
  (h5 : num_group2 = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((num_group1 : ℝ) * avg_age_group1 + (num_group2 : ℝ) * avg_age_group2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l1844_184490


namespace NUMINAMATH_CALUDE_difference_of_squares_l1844_184452

theorem difference_of_squares (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (product_eq : x * y = 99) :
  x^2 - y^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1844_184452


namespace NUMINAMATH_CALUDE_sum_set_size_bounds_l1844_184429

theorem sum_set_size_bounds (A : Finset ℕ) (S : Finset ℕ) : 
  A.card = 100 → 
  S = Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product A) → 
  199 ≤ S.card ∧ S.card ≤ 5050 := by
  sorry

end NUMINAMATH_CALUDE_sum_set_size_bounds_l1844_184429


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1844_184437

theorem smaller_root_of_quadratic (x : ℝ) :
  x^2 + 10*x - 24 = 0 → (x = -12 ∨ x = 2) ∧ -12 < 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1844_184437


namespace NUMINAMATH_CALUDE_dog_food_consumption_l1844_184463

/-- The amount of dog food one dog eats per day, in scoops -/
def dog_food_per_dog : ℝ := 0.12

/-- The number of dogs Ella owns -/
def number_of_dogs : ℕ := 2

/-- The total amount of dog food consumed by all dogs in a day, in scoops -/
def total_food_consumed : ℝ := dog_food_per_dog * number_of_dogs

theorem dog_food_consumption :
  total_food_consumed = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l1844_184463


namespace NUMINAMATH_CALUDE_bags_difference_l1844_184493

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 7

/-- The number of bags Tiffany found on the next day -/
def next_day_bags : ℕ := 12

/-- Theorem: The difference between the number of bags found on the next day
    and the number of bags on Monday is equal to 5 -/
theorem bags_difference : next_day_bags - monday_bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_bags_difference_l1844_184493


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_l1844_184411

def senate_committee_size : ℕ := 18
def num_republicans : ℕ := 10
def num_democrats : ℕ := 8
def subcommittee_republicans : ℕ := 4
def subcommittee_democrats : ℕ := 3

theorem subcommittee_formation_ways :
  (Nat.choose num_republicans subcommittee_republicans) *
  (Nat.choose num_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_l1844_184411


namespace NUMINAMATH_CALUDE_sum_smallest_largest_primes_l1844_184407

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primes_between (a b : ℕ) : Set ℕ :=
  {n : ℕ | a < n ∧ n < b ∧ is_prime n}

theorem sum_smallest_largest_primes :
  let P := primes_between 50 100
  ∃ (p q : ℕ), p ∈ P ∧ q ∈ P ∧
    (∀ x ∈ P, p ≤ x) ∧
    (∀ x ∈ P, x ≤ q) ∧
    p + q = 150 :=
sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_primes_l1844_184407


namespace NUMINAMATH_CALUDE_smallest_four_digit_geometric_even_l1844_184489

def is_geometric_sequence (a b c d : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

def digits_are_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_geometric_even :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    digits_are_distinct n ∧
    is_geometric_sequence (n / 1000) ((n / 100) % 10) ((n / 10) % 10) (n % 10) ∧
    Even n →
    n ≥ 1248 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_geometric_even_l1844_184489


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1844_184422

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 ∧ x ≠ 4 → x = 9 := by sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1844_184422


namespace NUMINAMATH_CALUDE_sector_max_area_angle_l1844_184472

/-- Given a sector with circumference 36, the radian measure of the central angle
    that maximizes the area of the sector is 2. -/
theorem sector_max_area_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  2 * r + l = 36 →
  α = l / r →
  (∀ r' l' α', 2 * r' + l' = 36 → α' = l' / r' →
    r * l ≥ r' * l') →
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_angle_l1844_184472


namespace NUMINAMATH_CALUDE_squirrel_nuts_collected_l1844_184471

/-- Represents the number of nuts eaten on day k -/
def nutsEatenOnDay (k : ℕ) : ℕ := k

/-- Represents the fraction of remaining nuts eaten each day -/
def fractionEaten : ℚ := 1 / 100

/-- Represents the number of nuts remaining before eating on day k -/
def nutsRemaining (k : ℕ) (totalNuts : ℕ) : ℕ :=
  totalNuts - (k - 1) * (k - 1 + 1) / 2

/-- Represents the number of nuts eaten on day k including the fraction -/
def totalNutsEatenOnDay (k : ℕ) (totalNuts : ℕ) : ℚ :=
  nutsEatenOnDay k + fractionEaten * (nutsRemaining k totalNuts - nutsEatenOnDay k)

/-- The theorem stating the total number of nuts collected by the squirrel -/
theorem squirrel_nuts_collected :
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, k < n → totalNutsEatenOnDay k 9801 < nutsRemaining k 9801) ∧
    nutsRemaining n 9801 = n :=
  sorry

end NUMINAMATH_CALUDE_squirrel_nuts_collected_l1844_184471


namespace NUMINAMATH_CALUDE_cookies_left_for_monica_l1844_184488

/-- The number of cookies Monica made for herself and her family. -/
def total_cookies : ℕ := 30

/-- The number of cookies Monica's father ate. -/
def father_cookies : ℕ := 10

/-- The number of cookies Monica's mother ate. -/
def mother_cookies : ℕ := father_cookies / 2

/-- The number of cookies Monica's brother ate. -/
def brother_cookies : ℕ := mother_cookies + 2

/-- Theorem stating the number of cookies left for Monica. -/
theorem cookies_left_for_monica : 
  total_cookies - father_cookies - mother_cookies - brother_cookies = 8 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_for_monica_l1844_184488


namespace NUMINAMATH_CALUDE_square_field_area_l1844_184476

/-- Given a square field where a horse takes 7 hours to run around it at a speed of 20 km/h,
    the area of the field is 1225 km². -/
theorem square_field_area (s : ℝ) (h : s > 0) : 
  (4 * s = 20 * 7) → s^2 = 1225 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l1844_184476


namespace NUMINAMATH_CALUDE_exists_composite_invariant_under_triplet_replacement_l1844_184477

/-- A function that replaces a triplet of digits at a given position in a natural number --/
def replaceTriplet (n : ℕ) (pos : ℕ) (newTriplet : ℕ) : ℕ :=
  sorry

/-- Predicate to check if a number is composite --/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

/-- The main theorem statement --/
theorem exists_composite_invariant_under_triplet_replacement :
  ∃ (N : ℕ), ∀ (pos : ℕ) (newTriplet : ℕ),
    isComposite (replaceTriplet N pos newTriplet) :=
  sorry

end NUMINAMATH_CALUDE_exists_composite_invariant_under_triplet_replacement_l1844_184477


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l1844_184426

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * total_average / (total_families - childless_families : ℚ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l1844_184426


namespace NUMINAMATH_CALUDE_students_left_on_bus_l1844_184494

def initial_students : ℕ := 10
def students_who_left : ℕ := 3

theorem students_left_on_bus : initial_students - students_who_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_left_on_bus_l1844_184494


namespace NUMINAMATH_CALUDE_age_ratio_l1844_184487

def sachin_age : ℕ := 14
def age_difference : ℕ := 7

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l1844_184487


namespace NUMINAMATH_CALUDE_jacqueline_erasers_l1844_184479

/-- The number of boxes of erasers Jacqueline has -/
def num_boxes : ℕ := 4

/-- The number of erasers in each box -/
def erasers_per_box : ℕ := 10

/-- The total number of erasers Jacqueline has -/
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 := by
  sorry

end NUMINAMATH_CALUDE_jacqueline_erasers_l1844_184479


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1844_184447

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/4) π) 
  (h2 : 3 * Real.cos (2 * α) = 4 * Real.sin (π/4 - α)) : 
  Real.sin (2 * α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1844_184447


namespace NUMINAMATH_CALUDE_asymptote_sum_l1844_184438

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) with integer coefficients A, B, C,
    if it has vertical asymptotes at x = -3, 0, and 4, then A + B + C = -13 -/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 4 → 
    x / (x^3 + A*x^2 + B*x + C) ≠ 0) →
  A + B + C = -13 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1844_184438


namespace NUMINAMATH_CALUDE_sum_of_values_l1844_184454

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  h_prob_sum : p₁ + p₂ = 1
  h_prob_nonneg : 0 ≤ p₁ ∧ 0 ≤ p₂

/-- Expected value of the discrete random variable -/
def expectation (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * X.p₂

/-- Variance of the discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expectation X)^2 + X.p₂ * (X.x₂ - expectation X)^2

theorem sum_of_values (X : DiscreteRV)
  (h_p₁ : X.p₁ = 2/3)
  (h_p₂ : X.p₂ = 1/3)
  (h_order : X.x₁ < X.x₂)
  (h_expectation : expectation X = 4/9)
  (h_variance : variance X = 2) :
  X.x₁ + X.x₂ = 17/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_values_l1844_184454


namespace NUMINAMATH_CALUDE_language_group_selection_l1844_184462

theorem language_group_selection (total : Nat) (english : Nat) (japanese : Nat)
  (h_total : total = 9)
  (h_english : english = 7)
  (h_japanese : japanese = 3)
  (h_at_least_one : english + japanese ≥ total) :
  (english * japanese) - (english + japanese - total) = 20 := by
  sorry

end NUMINAMATH_CALUDE_language_group_selection_l1844_184462


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l1844_184420

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_tangent_product (t : Triangle) 
  (h : t.a + t.c = 2 * t.b) : 
  Real.tan (t.A / 2) * Real.tan (t.C / 2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l1844_184420


namespace NUMINAMATH_CALUDE_max_quarters_is_19_l1844_184419

/-- Represents the number of each coin type in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Checks if the given coin count satisfies the problem conditions -/
def isValidCoinCount (c : CoinCount) : Prop :=
  c.nickels > 0 ∧ c.dimes > 0 ∧ c.quarters > 0 ∧
  c.nickels + c.dimes + c.quarters = 120 ∧
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters = 1000

/-- Theorem stating that 19 is the maximum number of quarters possible -/
theorem max_quarters_is_19 :
  ∀ c : CoinCount, isValidCoinCount c → c.quarters ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_is_19_l1844_184419


namespace NUMINAMATH_CALUDE_range_of_m_l1844_184415

-- Define the function f(x)
def f (x b c : ℝ) : ℝ := -2 * x^2 + b * x + c

-- State the theorem
theorem range_of_m (b c : ℝ) :
  (∀ x, f x b c > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x, -1 ≤ x ∧ x ≤ 0 → ∃ m, f x b c + m ≥ 4) →
  ∃ m₀, ∀ m, m ≥ m₀ ↔ (∀ x, -1 ≤ x ∧ x ≤ 0 → f x b c + m ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1844_184415


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l1844_184421

theorem irrational_among_given_numbers : 
  (¬ (∃ (a b : ℤ), (22 : ℚ) / 7 = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), (0.303003 : ℚ) = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), Real.sqrt 27 = a / b)) ∧ 
  (¬ (∃ (a b : ℤ), ((-64 : ℝ) ^ (1/3 : ℝ)) = a / b)) ↔ 
  (∃ (a b : ℤ), (22 : ℚ) / 7 = a / b) ∧ 
  (∃ (a b : ℤ), (0.303003 : ℚ) = a / b) ∧ 
  (¬ (∃ (a b : ℤ), Real.sqrt 27 = a / b)) ∧ 
  (∃ (a b : ℤ), ((-64 : ℝ) ^ (1/3 : ℝ)) = a / b) :=
sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l1844_184421


namespace NUMINAMATH_CALUDE_square_root_of_average_squares_ge_arithmetic_mean_l1844_184464

theorem square_root_of_average_squares_ge_arithmetic_mean
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_average_squares_ge_arithmetic_mean_l1844_184464


namespace NUMINAMATH_CALUDE_candy_distribution_l1844_184499

theorem candy_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2013 →
  a = 2 * b + 10 →
  a = 3 * c + 18 →
  a = 5 * d - 55 →
  a + b + c + d = total →
  a = 990 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1844_184499


namespace NUMINAMATH_CALUDE_cookie_radius_is_8_l1844_184496

/-- The equation of the cookie's boundary -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 21 = 4*x + 18*y

/-- The radius of the cookie -/
def cookie_radius : ℝ := 8

/-- Theorem stating that the radius of the cookie defined by the equation is 8 -/
theorem cookie_radius_is_8 :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = cookie_radius^2 :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_is_8_l1844_184496


namespace NUMINAMATH_CALUDE_book_length_l1844_184424

/-- The length of a rectangular book given its perimeter and width -/
theorem book_length (perimeter width : ℝ) (h1 : perimeter = 100) (h2 : width = 20) :
  2 * (width + (perimeter / 2 - width)) = perimeter :=
by
  sorry

#check book_length

end NUMINAMATH_CALUDE_book_length_l1844_184424


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l1844_184434

variable (P Q : Prop)

theorem equivalence_of_statements :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l1844_184434


namespace NUMINAMATH_CALUDE_triangle_theorem_l1844_184480

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * (1 + cos t.C) = t.c * (2 - cos t.B))
  (h2 : t.C = π / 3)
  (h3 : (1 / 2) * t.a * t.b * sin t.C = 4 * Real.sqrt 3) :
  (t.a + t.b = 2 * t.c) ∧ (t.c = 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1844_184480


namespace NUMINAMATH_CALUDE_base_eight_subtraction_l1844_184402

/-- Represents a number in base 8 -/
def BaseEight : Type := ℕ

/-- Converts a base 8 number to its decimal representation -/
def to_decimal (n : BaseEight) : ℕ := sorry

/-- Converts a decimal number to its base 8 representation -/
def to_base_eight (n : ℕ) : BaseEight := sorry

/-- Subtracts two base 8 numbers -/
def base_eight_sub (a b : BaseEight) : BaseEight := sorry

/-- Theorem stating that 46₈ - 27₈ = 17₈ in base 8 -/
theorem base_eight_subtraction :
  base_eight_sub (to_base_eight 38) (to_base_eight 23) = to_base_eight 15 := by sorry

end NUMINAMATH_CALUDE_base_eight_subtraction_l1844_184402


namespace NUMINAMATH_CALUDE_pictures_per_album_l1844_184465

theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 23) 
  (h2 : camera_pics = 7) 
  (h3 : num_albums = 5) 
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 6 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l1844_184465


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_six_l1844_184483

theorem sum_of_solutions_eq_six :
  ∃ (M₁ M₂ : ℝ), (M₁ * (M₁ - 6) = -5) ∧ (M₂ * (M₂ - 6) = -5) ∧ (M₁ + M₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_six_l1844_184483


namespace NUMINAMATH_CALUDE_living_room_to_bedroom_ratio_l1844_184413

/-- Energy usage of lights in Noah's house -/
def energy_usage (bedroom_watts_per_hour : ℝ) (hours : ℝ) (total_watts : ℝ) : Prop :=
  let bedroom_energy := bedroom_watts_per_hour * hours
  let office_energy := 3 * bedroom_energy
  let living_room_energy := total_watts - bedroom_energy - office_energy
  (living_room_energy / bedroom_energy = 4)

/-- Theorem: The ratio of living room light energy to bedroom light energy is 4:1 -/
theorem living_room_to_bedroom_ratio :
  energy_usage 6 2 96 := by
  sorry

end NUMINAMATH_CALUDE_living_room_to_bedroom_ratio_l1844_184413


namespace NUMINAMATH_CALUDE_all_circles_contain_common_point_l1844_184492

/-- A parabola of the form y = x² + 2px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The circle passing through the intersection points of a parabola with the coordinate axes -/
def circle_through_intersections (par : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | (x + par.p)^2 + (y - par.q/2)^2 = par.p^2 + par.q^2/4}

/-- Predicate to check if a parabola intersects the coordinate axes in three distinct points -/
def has_three_distinct_intersections (par : Parabola) : Prop :=
  par.p^2 > par.q ∧ par.q ≠ 0

theorem all_circles_contain_common_point :
  ∀ (par : Parabola), has_three_distinct_intersections par →
  (0, 1) ∈ circle_through_intersections par :=
sorry

end NUMINAMATH_CALUDE_all_circles_contain_common_point_l1844_184492


namespace NUMINAMATH_CALUDE_chair_distribution_count_l1844_184495

/-- The number of ways to distribute n identical objects into two groups,
    where one group must have at least a objects and the other group
    must have at least b objects. -/
def distribution_count (n a b : ℕ) : ℕ :=
  (n - a - b + 1).max 0

/-- Theorem: There are 5 ways to distribute 8 identical chairs into two groups,
    where one group (circle) must have at least 2 chairs and the other group (stack)
    must have at least 1 chair. -/
theorem chair_distribution_count : distribution_count 8 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chair_distribution_count_l1844_184495


namespace NUMINAMATH_CALUDE_dinner_time_calculation_l1844_184484

/-- Represents the time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the roasting time for a single turkey -/
def roastingTimePerTurkey (weight : ℕ) (minutesPerPound : ℕ) : ℕ :=
  weight * minutesPerPound

/-- Converts minutes to hours and minutes -/
def minutesToTime (totalMinutes : ℕ) : Time :=
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    valid := by sorry }

/-- Adds hours to a given time -/
def addHours (startTime : Time) (hoursToAdd : ℕ) : Time :=
  { hours := (startTime.hours + hoursToAdd) % 24,
    minutes := startTime.minutes,
    valid := startTime.valid }

/-- Calculates the dinner time given the conditions -/
def calculateDinnerTime (numTurkeys : ℕ) (turkeyWeight : ℕ) (minutesPerPound : ℕ) (startTime : Time) : Time :=
  let totalRoastingMinutes := numTurkeys * roastingTimePerTurkey turkeyWeight minutesPerPound
  let totalRoastingHours := totalRoastingMinutes / 60
  addHours startTime totalRoastingHours

theorem dinner_time_calculation :
  let numTurkeys : ℕ := 2
  let turkeyWeight : ℕ := 16
  let minutesPerPound : ℕ := 15
  let startTime : Time := { hours := 10, minutes := 0, valid := by sorry }
  calculateDinnerTime numTurkeys turkeyWeight minutesPerPound startTime =
    { hours := 18, minutes := 0, valid := by sorry } := by sorry


end NUMINAMATH_CALUDE_dinner_time_calculation_l1844_184484


namespace NUMINAMATH_CALUDE_domino_coverage_l1844_184400

theorem domino_coverage (n k : ℕ+) :
  (∃ (coverage : Fin n × Fin n → Fin k × Bool),
    (∀ (i j : Fin n), ∃ (x : Fin k) (b : Bool),
      coverage (i, j) = (x, b) ∧
      (b = true → coverage (i, j.succ) = (x, false)) ∧
      (b = false → coverage (i.succ, j) = (x, true))))
  ↔ k ∣ n := by sorry

end NUMINAMATH_CALUDE_domino_coverage_l1844_184400


namespace NUMINAMATH_CALUDE_compare_expressions_l1844_184450

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 / b + b^2 / a) > (a + b) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l1844_184450


namespace NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_three_quarters_l1844_184470

theorem mean_of_five_numbers_with_sum_three_quarters
  (a b c d e : ℝ) (h : a + b + c + d + e = 3/4) :
  (a + b + c + d + e) / 5 = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_numbers_with_sum_three_quarters_l1844_184470


namespace NUMINAMATH_CALUDE_archipelago_islands_l1844_184459

theorem archipelago_islands (n : ℕ) : 
  (n * (n - 1)) / 2 + n = 28 →
  n + 1 = 8 :=
by
  sorry

#check archipelago_islands

end NUMINAMATH_CALUDE_archipelago_islands_l1844_184459


namespace NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_15_l1844_184406

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2*a
def F (a b : ℝ) : ℝ := b^2 + a*b

-- State the theorem
theorem F_of_2_f_of_3_equals_15 : F 2 (f 3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_F_of_2_f_of_3_equals_15_l1844_184406


namespace NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l1844_184439

theorem half_plus_five_equals_fifteen (n : ℝ) : (1/2) * n + 5 = 15 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_fifteen_l1844_184439


namespace NUMINAMATH_CALUDE_division_problem_l1844_184460

theorem division_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1636 → 
  L = Q * S + 10 → 
  Q = 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1844_184460


namespace NUMINAMATH_CALUDE_possible_values_of_b_over_a_l1844_184433

theorem possible_values_of_b_over_a (a b : ℝ) (h : a > 0) :
  (∀ a b, a > 0 → Real.log a + b - a * Real.exp (b - 1) ≥ 0) →
  (b / a = Real.exp (-1) ∨ b / a = Real.exp (-2) ∨ b / a = -Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_b_over_a_l1844_184433


namespace NUMINAMATH_CALUDE_railway_optimization_l1844_184445

/-- The number of round trips per day as a function of the number of carriages -/
def t (n : ℕ) : ℤ := -2 * n + 24

/-- The number of passengers per day as a function of the number of carriages -/
def y (n : ℕ) : ℤ := t n * n * 110 * 2

theorem railway_optimization :
  (t 4 = 16 ∧ t 7 = 10) ∧ 
  (∀ n : ℕ, 1 ≤ n → n < 12 → y n ≤ y 6) ∧
  y 6 = 15840 := by
  sorry

#eval t 4  -- Expected: 16
#eval t 7  -- Expected: 10
#eval y 6  -- Expected: 15840

end NUMINAMATH_CALUDE_railway_optimization_l1844_184445


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1844_184475

theorem quadratic_factorization (C D E F : ℤ) :
  (∀ y : ℝ, 10 * y^2 - 51 * y + 21 = (C * y - D) * (E * y - F)) →
  C * E + C = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1844_184475


namespace NUMINAMATH_CALUDE_brothers_combined_age_l1844_184444

/-- Given the ages of Michael and his three brothers, prove their combined age is 53 years. -/
theorem brothers_combined_age :
  ∀ (michael oldest older younger : ℕ),
  -- The oldest brother is 1 year older than twice Michael's age when Michael was a year younger
  oldest = 2 * (michael - 1) + 1 →
  -- The younger brother is 5 years old
  younger = 5 →
  -- The younger brother's age is a third of the older brother's age
  older = 3 * younger →
  -- The other brother is half the age of the oldest brother
  older = oldest / 2 →
  -- The other brother is three years younger than Michael
  older = michael - 3 →
  -- The other brother is twice as old as their youngest brother
  older = 2 * younger →
  -- The combined age of all four brothers is 53
  michael + oldest + older + younger = 53 := by
sorry


end NUMINAMATH_CALUDE_brothers_combined_age_l1844_184444


namespace NUMINAMATH_CALUDE_monomial_satisfies_conditions_l1844_184456

-- Define a structure for monomials
structure Monomial (α : Type) [CommRing α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the monomial -2mn^2
def target_monomial : Monomial ℤ := ⟨-2, [(1, 1), (2, 2)]⟩

-- Define functions to check the conditions
def has_variables (m : Monomial ℤ) (vars : List Nat) : Prop :=
  ∀ v ∈ vars, ∃ p ∈ m.vars, v = p.1

def coefficient (m : Monomial ℤ) : ℤ := m.coeff

def degree (m : Monomial ℤ) : Nat :=
  m.vars.foldr (fun p acc => acc + p.2) 0

-- Theorem statement
theorem monomial_satisfies_conditions :
  has_variables target_monomial [1, 2] ∧
  coefficient target_monomial = -2 ∧
  degree target_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_satisfies_conditions_l1844_184456


namespace NUMINAMATH_CALUDE_set_equality_l1844_184498

def U : Set ℕ := Set.univ

def A : Set ℕ := {x | ∃ n : ℕ, x = 2 * n}

def B : Set ℕ := {x | ∃ n : ℕ, x = 4 * n}

theorem set_equality : U = A ∪ (U \ B) := by sorry

end NUMINAMATH_CALUDE_set_equality_l1844_184498


namespace NUMINAMATH_CALUDE_line_slope_through_point_l1844_184491

theorem line_slope_through_point (x y k : ℝ) : 
  x = 2 → y = Real.sqrt 3 → y = k * x → k = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_through_point_l1844_184491


namespace NUMINAMATH_CALUDE_closest_point_l1844_184414

def v (t : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2 + 5*t
  | 1 => -3 + 7*t
  | 2 => -3 - 2*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => 4
  | 2 => 5

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 7
  | 2 => -2

theorem closest_point :
  let t := 43 / 78
  (v t - a) • direction = 0 ∧
  ∀ s, s ≠ t → ‖v s - a‖ > ‖v t - a‖ :=
sorry

end NUMINAMATH_CALUDE_closest_point_l1844_184414


namespace NUMINAMATH_CALUDE_mutual_fund_change_l1844_184425

theorem mutual_fund_change (initial_value : ℝ) (h : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  let percent_change := (day2_value - initial_value) / initial_value * 100
  percent_change = 5 := by sorry

end NUMINAMATH_CALUDE_mutual_fund_change_l1844_184425


namespace NUMINAMATH_CALUDE_rectangle_area_l1844_184486

/-- The area of a rectangle with given vertices in a rectangular coordinate system -/
theorem rectangle_area (a b c d : ℝ × ℝ) : 
  a = (-3, 1) → b = (1, 1) → c = (1, -2) → d = (-3, -2) →
  (b.1 - a.1) * (a.2 - d.2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1844_184486


namespace NUMINAMATH_CALUDE_planes_perpendicular_l1844_184412

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel, perpendicular, and subset relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular n β)
  (h3 : subset m α) :
  perp_planes α β := by sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l1844_184412


namespace NUMINAMATH_CALUDE_systematic_sampling_l1844_184446

theorem systematic_sampling (n : Nat) (groups : Nat) (last_group_num : Nat) :
  n = 100 ∧ groups = 5 ∧ last_group_num = 94 →
  ∃ (interval : Nat) (first_group_num : Nat),
    interval * (groups - 1) + first_group_num = last_group_num ∧
    interval * 1 + first_group_num = 34 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1844_184446


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l1844_184432

/-- Given a rectangle with dimensions 6 x 9, prove that the ratio of the volume of the larger cylinder
    to the volume of the smaller cylinder formed by rolling the rectangle is 3/2. -/
theorem rectangle_cylinder_volume_ratio :
  let width : ℝ := 6
  let length : ℝ := 9
  let volume1 : ℝ := π * (width / (2 * π))^2 * length
  let volume2 : ℝ := π * (length / (2 * π))^2 * width
  volume2 / volume1 = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l1844_184432


namespace NUMINAMATH_CALUDE_max_money_is_zero_l1844_184461

/-- Represents the state of the stone piles and A's money --/
structure GameState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  money : ℤ

/-- Represents a move from one pile to another --/
inductive Move
  | one_to_two
  | one_to_three
  | two_to_one
  | two_to_three
  | three_to_one
  | three_to_two

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.one_to_two => 
      { pile1 := state.pile1 - 1, 
        pile2 := state.pile2 + 1, 
        pile3 := state.pile3,
        money := state.money + (state.pile2 - state.pile1 + 1) }
  | Move.one_to_three => 
      { pile1 := state.pile1 - 1, 
        pile2 := state.pile2, 
        pile3 := state.pile3 + 1,
        money := state.money + (state.pile3 - state.pile1 + 1) }
  | Move.two_to_one => 
      { pile1 := state.pile1 + 1, 
        pile2 := state.pile2 - 1, 
        pile3 := state.pile3,
        money := state.money + (state.pile1 - state.pile2 + 1) }
  | Move.two_to_three => 
      { pile1 := state.pile1, 
        pile2 := state.pile2 - 1, 
        pile3 := state.pile3 + 1,
        money := state.money + (state.pile3 - state.pile2 + 1) }
  | Move.three_to_one => 
      { pile1 := state.pile1 + 1, 
        pile2 := state.pile2, 
        pile3 := state.pile3 - 1,
        money := state.money + (state.pile1 - state.pile3 + 1) }
  | Move.three_to_two => 
      { pile1 := state.pile1, 
        pile2 := state.pile2 + 1, 
        pile3 := state.pile3 - 1,
        money := state.money + (state.pile2 - state.pile3 + 1) }

/-- Theorem: The maximum amount of money A can have when all stones return to their initial positions is 0 --/
theorem max_money_is_zero (initial : GameState) (moves : List Move) :
  (moves.foldl applyMove initial).pile1 = initial.pile1 ∧
  (moves.foldl applyMove initial).pile2 = initial.pile2 ∧
  (moves.foldl applyMove initial).pile3 = initial.pile3 →
  (moves.foldl applyMove initial).money ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_money_is_zero_l1844_184461


namespace NUMINAMATH_CALUDE_max_ticket_types_for_specific_car_l1844_184451

/-- Represents a one-way traveling car with stations and capacity. -/
structure TravelingCar where
  num_stations : Nat
  capacity : Nat

/-- Calculates the maximum number of different ticket types that can be sold. -/
def max_ticket_types (car : TravelingCar) : Nat :=
  let total_possible_tickets := (car.num_stations - 1) * car.num_stations / 2
  let max_non_overlapping_tickets := ((car.num_stations + 1) / 2) ^ 2
  let unsellable_tickets := max_non_overlapping_tickets - car.capacity
  total_possible_tickets - unsellable_tickets

/-- Theorem stating the maximum number of different ticket types for a specific car configuration. -/
theorem max_ticket_types_for_specific_car :
  let car := TravelingCar.mk 14 25
  max_ticket_types car = 67 := by
  sorry

end NUMINAMATH_CALUDE_max_ticket_types_for_specific_car_l1844_184451


namespace NUMINAMATH_CALUDE_cube_order_l1844_184428

theorem cube_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_l1844_184428


namespace NUMINAMATH_CALUDE_smallest_odd_four_prime_factors_l1844_184469

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_exactly_four_prime_factors (n : ℕ) : Prop :=
  ∃ (p q r s : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    n = p * q * r * s

theorem smallest_odd_four_prime_factors :
  (1155 % 2 = 1) ∧
  has_exactly_four_prime_factors 1155 ∧
  ∀ n : ℕ, n < 1155 → (n % 2 = 1 → ¬has_exactly_four_prime_factors n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_four_prime_factors_l1844_184469


namespace NUMINAMATH_CALUDE_factor_implies_q_value_l1844_184455

theorem factor_implies_q_value (q : ℚ) :
  (∀ m : ℚ, (m - 8) ∣ (m^2 - q*m - 24)) → q = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_q_value_l1844_184455


namespace NUMINAMATH_CALUDE_equation_system_result_l1844_184418

theorem equation_system_result (x y z : ℝ) 
  (eq1 : 2*x + y + z = 6) 
  (eq2 : x + 2*y + z = 7) : 
  5*x^2 + 8*x*y + 5*y^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_equation_system_result_l1844_184418


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1844_184473

/-- The number of ways to arrange books of different languages on a shelf --/
def arrange_books (total : ℕ) (italian : ℕ) (german : ℕ) (french : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial italian * Nat.factorial german * Nat.factorial french

/-- Theorem stating the number of arrangements for the given book problem --/
theorem book_arrangement_count :
  arrange_books 11 3 3 5 = 25920 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1844_184473


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1844_184443

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (d : ℕ) : List (Fin 4) := sorry

theorem binary_to_quaternary_conversion :
  let binary : List Bool := [true, true, false, true, true, false, true, true, false, true]
  let quaternary : List (Fin 4) := [3, 1, 1, 3, 1]
  binary_to_decimal binary = (quaternary.map (λ x => x.val)).foldl (λ acc x => acc * 4 + x) 0 :=
by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1844_184443


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_x_in_range_l1844_184441

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - 1| + |x - 2*a|

-- Part I
theorem solution_set_when_a_is_one :
  let a := 1
  ∀ x, f x a ≤ 3 ↔ x ∈ Set.Icc 0 2 :=
sorry

-- Part II
theorem a_value_when_x_in_range :
  (∀ x ∈ Set.Icc 1 2, f x a ≤ 3) → a = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_x_in_range_l1844_184441


namespace NUMINAMATH_CALUDE_average_of_list_l1844_184409

def number_list : List Nat := [55, 48, 507, 2, 684, 42]

theorem average_of_list (list : List Nat) : 
  (list.sum / list.length : ℚ) = 223 :=
by sorry

end NUMINAMATH_CALUDE_average_of_list_l1844_184409


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1844_184401

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x > 2 ∧ (x - 1)^2 > 1) ∧
  (∃ x : ℝ, (x - 1)^2 > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1844_184401


namespace NUMINAMATH_CALUDE_min_visible_sum_l1844_184466

/-- Represents a die in the cube -/
structure Die where
  sides : Fin 6 → ℕ
  opposite_sum : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents the 4x4x4 cube made of dice -/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visible_sum (c : Cube) : ℕ := sorry

/-- The theorem stating the minimum possible visible sum -/
theorem min_visible_sum (c : Cube) : visible_sum c ≥ 144 := by sorry

end NUMINAMATH_CALUDE_min_visible_sum_l1844_184466


namespace NUMINAMATH_CALUDE_smith_family_seating_arrangement_l1844_184468

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smith_family_seating_arrangement :
  let total_arrangements := factorial 7
  let no_adjacent_boys := factorial 4 * factorial 3
  total_arrangements - no_adjacent_boys = 4896 :=
by sorry

end NUMINAMATH_CALUDE_smith_family_seating_arrangement_l1844_184468


namespace NUMINAMATH_CALUDE_no_distinct_perfect_squares_sum_to_100_l1844_184474

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def distinct_perfect_squares_sum_to_100 : Prop :=
  ∃ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
    a + b + c = 100

theorem no_distinct_perfect_squares_sum_to_100 : ¬distinct_perfect_squares_sum_to_100 := by
  sorry

end NUMINAMATH_CALUDE_no_distinct_perfect_squares_sum_to_100_l1844_184474


namespace NUMINAMATH_CALUDE_outstanding_student_distribution_count_l1844_184481

/-- The number of ways to distribute n items among k groups, with each group receiving at least one item. -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The number of distribution plans for 10 "Outstanding Student" spots among 6 classes, with each class receiving at least one spot. -/
def outstanding_student_distribution : ℕ := distribute 10 6

theorem outstanding_student_distribution_count : outstanding_student_distribution = 126 := by
  sorry

end NUMINAMATH_CALUDE_outstanding_student_distribution_count_l1844_184481


namespace NUMINAMATH_CALUDE_parallel_tangents_intersection_l1844_184442

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (∃ (k : ℝ), (2 * x₀ = k) ∧ (-3 * x₀^2 = k)) → (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_tangents_intersection_l1844_184442


namespace NUMINAMATH_CALUDE_two_digit_sum_l1844_184405

theorem two_digit_sum (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → a ≠ 0 → a - b = a * b → 
  10 * a + b + (10 * b + a) = 22 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_l1844_184405


namespace NUMINAMATH_CALUDE_angle_relations_l1844_184449

theorem angle_relations (θ : Real) 
  (h1 : θ ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) -- θ is in the fourth quadrant
  (h2 : Real.sin θ + Real.cos θ = 1/5) :
  (Real.sin θ - Real.cos θ = -7/5) ∧ (Real.tan θ = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_angle_relations_l1844_184449


namespace NUMINAMATH_CALUDE_probability_no_adjacent_same_l1844_184403

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of sides on the die -/
def sides : ℕ := 6

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := 25 / 108

/-- Theorem stating the probability of no two adjacent people rolling the same number -/
theorem probability_no_adjacent_same :
  prob_no_adjacent_same = 25 / 108 := by sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_same_l1844_184403


namespace NUMINAMATH_CALUDE_factors_of_48_l1844_184453

/-- The number of distinct positive factors of 48 is 10. -/
theorem factors_of_48 : Finset.card (Nat.divisors 48) = 10 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_48_l1844_184453


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1844_184448

theorem lcm_hcf_problem (a b : ℕ+) (h1 : Nat.lcm a b = 25974) (h2 : Nat.gcd a b = 107) (h3 : a = 4951) : b = 561 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1844_184448


namespace NUMINAMATH_CALUDE_max_value_problem_l1844_184410

theorem max_value_problem (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l1844_184410


namespace NUMINAMATH_CALUDE_hex_B1F_to_dec_l1844_184440

def hex_to_dec (hex : String) : ℕ :=
  hex.foldr (fun c acc => 16 * acc + 
    match c with
    | 'A' => 10
    | 'B' => 11
    | 'C' => 12
    | 'D' => 13
    | 'E' => 14
    | 'F' => 15
    | _ => c.toNat - '0'.toNat
  ) 0

theorem hex_B1F_to_dec : hex_to_dec "B1F" = 2847 := by
  sorry

end NUMINAMATH_CALUDE_hex_B1F_to_dec_l1844_184440


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1844_184435

theorem necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1844_184435


namespace NUMINAMATH_CALUDE_remainder_of_12345678910_mod_101_l1844_184430

theorem remainder_of_12345678910_mod_101 : 12345678910 % 101 = 31 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_12345678910_mod_101_l1844_184430


namespace NUMINAMATH_CALUDE_coffee_mix_price_l1844_184458

/-- The price of the first kind of coffee in dollars per pound -/
def price_first : ℚ := 215 / 100

/-- The price of the mixed coffee in dollars per pound -/
def price_mix : ℚ := 230 / 100

/-- The total weight of the mixed coffee in pounds -/
def total_weight : ℚ := 18

/-- The weight of each kind of coffee in the mix in pounds -/
def weight_each : ℚ := 9

/-- The price of the second kind of coffee in dollars per pound -/
def price_second : ℚ := 245 / 100

theorem coffee_mix_price :
  price_second = 
    (price_mix * total_weight - price_first * weight_each) / weight_each :=
by sorry

end NUMINAMATH_CALUDE_coffee_mix_price_l1844_184458


namespace NUMINAMATH_CALUDE_garden_shorter_side_l1844_184482

theorem garden_shorter_side (perimeter : ℝ) (area : ℝ) : perimeter = 60 ∧ area = 200 → ∃ x y : ℝ, x ≤ y ∧ 2*x + 2*y = perimeter ∧ x*y = area ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_shorter_side_l1844_184482


namespace NUMINAMATH_CALUDE_constant_term_proof_l1844_184485

/-- The constant term in the expansion of (x^2 + 2/x^3)^5 -/
def constant_term : ℕ := 40

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem constant_term_proof :
  constant_term = binomial 5 2 * 2^2 :=
sorry

end NUMINAMATH_CALUDE_constant_term_proof_l1844_184485


namespace NUMINAMATH_CALUDE_particle_speed_at_2_l1844_184427

/-- The position of a particle at time t -/
def particle_position (t : ℝ) : ℝ × ℝ :=
  (t^2 + 2*t + 7, 3*t^2 + 4*t - 13)

/-- The speed of the particle at time t -/
noncomputable def particle_speed (t : ℝ) : ℝ :=
  let pos_t := particle_position t
  let pos_next := particle_position (t + 1)
  let dx := pos_next.1 - pos_t.1
  let dy := pos_next.2 - pos_t.2
  Real.sqrt (dx^2 + dy^2)

/-- Theorem: The speed of the particle at t = 2 is √410 -/
theorem particle_speed_at_2 :
  particle_speed 2 = Real.sqrt 410 := by
  sorry

end NUMINAMATH_CALUDE_particle_speed_at_2_l1844_184427
