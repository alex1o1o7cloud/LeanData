import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_average_l858_85804

theorem consecutive_odd_numbers_average (a b c d : ℕ) : 
  a = 27 ∧ 
  b = a - 2 ∧ 
  c = b - 2 ∧ 
  d = c - 2 ∧ 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d → 
  (a + b + c + d) / 4 = 24 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_average_l858_85804


namespace NUMINAMATH_CALUDE_sum_of_parameters_l858_85830

/-- Given two quadratic equations with solution sets M and N,
    prove that the sum of their parameters is 21 when their intersection is {2}. -/
theorem sum_of_parameters (p q : ℝ) : 
  (∃ M : Set ℝ, ∀ x ∈ M, x^2 - p*x + 6 = 0) →
  (∃ N : Set ℝ, ∀ x ∈ N, x^2 + 6*x - q = 0) →
  (∃ M N : Set ℝ, 
    (∀ x ∈ M, x^2 - p*x + 6 = 0) ∧ 
    (∀ x ∈ N, x^2 + 6*x - q = 0) ∧ 
    (M ∩ N = {2})) →
  p + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parameters_l858_85830


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l858_85825

theorem three_digit_number_proof :
  ∀ a b c : ℕ,
  (100 ≤ a * 100 + b * 10 + c) → 
  (a * 100 + b * 10 + c < 1000) →
  (a * (b + c) = 33) →
  (b * (a + c) = 40) →
  (a * 100 + b * 10 + c = 347) :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l858_85825


namespace NUMINAMATH_CALUDE_locus_of_centroid_l858_85889

/-- The locus of the centroid of a triangle formed by specific points on a line and parabola -/
theorem locus_of_centroid (k : ℝ) (x y : ℝ) : 
  -- Line l: y = k(x - 2)
  (∀ x, y = k * (x - 2)) →
  -- Parabola: y = x^2 + 2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ k * (x₁ - 2) = x₁^2 + 2 ∧ k * (x₂ - 2) = x₂^2 + 2) →
  -- Conditions on k
  (k < 4 - 2 * Real.sqrt 6 ∨ k > 4 + 2 * Real.sqrt 6) →
  k ≠ 0 →
  -- Point P conditions
  (∃ x₀ y₀, y₀ = (12 * k) / (k - 4) ∧ x₀ = 12 / (k - 4) + 2) →
  -- Centroid G(x, y)
  (x = (4 / (k - 4)) + 4/3 ∧ y = (4 * k) / (k - 4)) →
  -- Locus equation
  12 * x - 3 * y - 4 = 0 ∧ 
  4 - (4/3) * Real.sqrt 6 < y ∧ 
  y < 4 + (4/3) * Real.sqrt 6 ∧ 
  y ≠ 4 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_centroid_l858_85889


namespace NUMINAMATH_CALUDE_opposite_of_four_l858_85851

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- The opposite of 4 is -4. -/
theorem opposite_of_four : opposite 4 = -4 := by sorry

end NUMINAMATH_CALUDE_opposite_of_four_l858_85851


namespace NUMINAMATH_CALUDE_m_div_60_eq_483840_l858_85864

/-- The smallest positive integer that is a multiple of 60 and has exactly 96 positive integral divisors -/
def m : ℕ := sorry

/-- m is a multiple of 60 -/
axiom m_multiple_of_60 : 60 ∣ m

/-- m has exactly 96 positive integral divisors -/
axiom m_divisors_count : (Finset.filter (· ∣ m) (Finset.range m)).card = 96

/-- m is the smallest such number -/
axiom m_smallest : ∀ k : ℕ, k < m → ¬(60 ∣ k ∧ (Finset.filter (· ∣ k) (Finset.range k)).card = 96)

/-- The main theorem -/
theorem m_div_60_eq_483840 : m / 60 = 483840 := sorry

end NUMINAMATH_CALUDE_m_div_60_eq_483840_l858_85864


namespace NUMINAMATH_CALUDE_cut_difference_l858_85817

/-- The amount cut off the skirt in inches -/
def skirt_cut : ℝ := 0.75

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The difference between the amount cut off the skirt and the amount cut off the pants -/
theorem cut_difference : skirt_cut - pants_cut = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_cut_difference_l858_85817


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l858_85852

theorem triangle_max_perimeter :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    b = 4 * a ∧ 
    a + b > 18 ∧ 
    a + 18 > b ∧ 
    b + 18 > a ∧
    ∀ (x y : ℕ), 
      x > 0 → 
      y > 0 → 
      y = 4 * x → 
      x + y > 18 → 
      x + 18 > y → 
      y + 18 > x → 
      a + b + 18 ≥ x + y + 18 ∧
    a + b + 18 = 43 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l858_85852


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l858_85870

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y 
  (h1 : ∃ (k : ℚ), w = (5/2) * k ∧ x = k) 
  (h2 : ∃ (m : ℚ), y = 4 * m ∧ z = m) 
  (h3 : ∃ (n : ℚ), z = (1/8) * n ∧ x = n) : 
  w = 5 * y := by sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l858_85870


namespace NUMINAMATH_CALUDE_square_difference_equals_square_l858_85874

theorem square_difference_equals_square (x : ℝ) : (10 - x)^2 = x^2 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_square_l858_85874


namespace NUMINAMATH_CALUDE_awards_distribution_l858_85884

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The condition that each student receives at least one award -/
def at_least_one_award (distribution : List ℕ) : Prop :=
  sorry

theorem awards_distribution :
  ∃ (d : List ℕ),
    d.length = 4 ∧
    d.sum = 6 ∧
    at_least_one_award d ∧
    distribute_awards 6 4 = 1560 :=
  sorry

end NUMINAMATH_CALUDE_awards_distribution_l858_85884


namespace NUMINAMATH_CALUDE_father_age_proof_l858_85865

/-- The age of the father -/
def father_age : ℕ := 48

/-- The age of the son -/
def son_age : ℕ := 75 - father_age

/-- The time difference between when the father was the son's current age and now -/
def time_difference : ℕ := father_age - son_age

theorem father_age_proof :
  (father_age + son_age = 75) ∧
  (father_age = 8 * (son_age - time_difference)) ∧
  (father_age - time_difference = son_age) →
  father_age = 48 :=
by sorry

end NUMINAMATH_CALUDE_father_age_proof_l858_85865


namespace NUMINAMATH_CALUDE_fraction_sum_equals_123_128th_l858_85896

theorem fraction_sum_equals_123_128th : 
  (4 : ℚ) / 4 + 7 / 8 + 12 / 16 + 19 / 32 + 28 / 64 + 39 / 128 - 3 = 123 / 128 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_123_128th_l858_85896


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l858_85868

/-- Given an ellipse and a hyperbola with coinciding foci, prove that b² of the ellipse equals 14.76 -/
theorem ellipse_hyperbola_coinciding_foci (b : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/b^2 = 1 → x^2/100 - y^2/64 = 1/16 → 
   ∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 10.25) → 
  b^2 = 14.76 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l858_85868


namespace NUMINAMATH_CALUDE_product_of_two_positive_quantities_l858_85814

theorem product_of_two_positive_quantities (s : ℝ) (h : s > 0) :
  ¬(∀ x : ℝ, 0 < x → x < s → 
    (x * (s - x) ≤ y * (s - y) → (x = 0 ∨ x = s))) :=
sorry

end NUMINAMATH_CALUDE_product_of_two_positive_quantities_l858_85814


namespace NUMINAMATH_CALUDE_probability_of_three_common_books_l858_85806

theorem probability_of_three_common_books :
  let total_books : ℕ := 12
  let books_per_student : ℕ := 6
  let common_books : ℕ := 3
  
  let total_outcomes : ℕ := (Nat.choose total_books books_per_student) ^ 2
  let successful_outcomes : ℕ := 
    (Nat.choose total_books common_books) * 
    (Nat.choose (total_books - common_books) (books_per_student - common_books)) * 
    (Nat.choose (total_books - books_per_student) (books_per_student - common_books))
  
  (successful_outcomes : ℚ) / total_outcomes = 5 / 23
  := by sorry

end NUMINAMATH_CALUDE_probability_of_three_common_books_l858_85806


namespace NUMINAMATH_CALUDE_matrix_transformation_and_eigenvalues_l858_85876

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 2, 2]

theorem matrix_transformation_and_eigenvalues :
  -- 1) A transforms (1, 2) to (7, 6)
  A.mulVec ![1, 2] = ![7, 6] ∧
  -- 2) The eigenvalues of A are -1 and 4
  (A.charpoly.roots.toFinset = {-1, 4}) ∧
  -- 3) [3, -2] is an eigenvector for λ = -1
  (A.mulVec ![3, -2] = (-1 : ℝ) • ![3, -2]) ∧
  -- 4) [1, 1] is an eigenvector for λ = 4
  (A.mulVec ![1, 1] = (4 : ℝ) • ![1, 1]) := by
sorry


end NUMINAMATH_CALUDE_matrix_transformation_and_eigenvalues_l858_85876


namespace NUMINAMATH_CALUDE_translation_right_4_units_l858_85802

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in the x-direction -/
def translateX (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

theorem translation_right_4_units (P : Point) (P' : Point) :
  P.x = -2 ∧ P.y = 3 →
  P' = translateX P 4 →
  P'.x = 2 ∧ P'.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_4_units_l858_85802


namespace NUMINAMATH_CALUDE_range_of_a_l858_85816

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x + 3 ≤ a^2 - 2*a - 1)) → 
  (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l858_85816


namespace NUMINAMATH_CALUDE_fraction_integer_iff_q_values_l858_85801

theorem fraction_integer_iff_q_values (q : ℕ+) :
  (∃ (k : ℕ+), (4 * q + 28 : ℚ) / (3 * q - 7 : ℚ) = k) ↔ q ∈ ({7, 15, 25} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_q_values_l858_85801


namespace NUMINAMATH_CALUDE_product_expansion_l858_85822

theorem product_expansion (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l858_85822


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l858_85863

theorem arithmetic_sequence_sum_ratio (a₁ d : ℚ) : 
  let S : ℕ → ℚ := λ n => n * a₁ + n * (n - 1) / 2 * d
  (S 3) / (S 7) = 1 / 3 → (S 6) / (S 7) = 17 / 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l858_85863


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l858_85844

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) : ℕ :=
  let cans_for_children := (children_fed + can_capacity.children - 1) / can_capacity.children
  let remaining_cans := total_cans - cans_for_children
  remaining_cans * can_capacity.adults

/-- The main theorem to be proved -/
theorem soup_feeding_theorem (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) :
  total_cans = 8 →
  can_capacity.adults = 5 →
  can_capacity.children = 10 →
  children_fed = 20 →
  remaining_adults_fed total_cans can_capacity children_fed = 30 := by
  sorry

#check soup_feeding_theorem

end NUMINAMATH_CALUDE_soup_feeding_theorem_l858_85844


namespace NUMINAMATH_CALUDE_urea_formation_moles_l858_85858

-- Define the chemical species
inductive ChemicalSpecies
| CarbonDioxide
| Ammonia
| Urea
| Water

-- Define a structure for chemical reactions
structure ChemicalReaction where
  reactants : List (ChemicalSpecies × ℚ)
  products : List (ChemicalSpecies × ℚ)

-- Define the urea formation reaction
def ureaFormationReaction : ChemicalReaction :=
  { reactants := [(ChemicalSpecies.CarbonDioxide, 1), (ChemicalSpecies.Ammonia, 2)]
  , products := [(ChemicalSpecies.Urea, 1), (ChemicalSpecies.Water, 1)] }

-- Define a function to calculate the moles of product formed
def molesOfProductFormed (reaction : ChemicalReaction) (limitingReactant : ChemicalSpecies) (molesOfLimitingReactant : ℚ) (product : ChemicalSpecies) : ℚ :=
  sorry -- Implementation details omitted

-- Theorem statement
theorem urea_formation_moles :
  molesOfProductFormed ureaFormationReaction ChemicalSpecies.CarbonDioxide 1 ChemicalSpecies.Urea = 1 :=
sorry

end NUMINAMATH_CALUDE_urea_formation_moles_l858_85858


namespace NUMINAMATH_CALUDE_min_value_quadratic_l858_85834

theorem min_value_quadratic (x y : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l858_85834


namespace NUMINAMATH_CALUDE_two_heads_probability_l858_85807

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the outcome of tossing two coins simultaneously -/
def TwoCoinsOutcome := (CoinOutcome × CoinOutcome)

/-- The set of all possible outcomes when tossing two coins -/
def allOutcomes : Finset TwoCoinsOutcome := sorry

/-- The set of outcomes where both coins show heads -/
def twoHeadsOutcomes : Finset TwoCoinsOutcome := sorry

/-- Proposition: The probability of getting two heads when tossing two fair coins simultaneously is 1/4 -/
theorem two_heads_probability :
  (Finset.card twoHeadsOutcomes) / (Finset.card allOutcomes : ℚ) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_two_heads_probability_l858_85807


namespace NUMINAMATH_CALUDE_exactly_three_correct_deliveries_l858_85875

def n : ℕ := 5

-- The probability of exactly k successes in n trials
def probability (k : ℕ) : ℚ :=
  (n.choose k * (n - k).factorial) / n.factorial

-- The main theorem
theorem exactly_three_correct_deliveries : probability 3 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_correct_deliveries_l858_85875


namespace NUMINAMATH_CALUDE_f_range_l858_85836

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 4 * Real.sin x + 2 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem f_range :
  Set.range (fun (x : ℝ) => f x) = Set.Icc (-12) 0 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l858_85836


namespace NUMINAMATH_CALUDE_trevor_eggs_left_l858_85850

/-- Represents the number of eggs laid by each chicken and the number of eggs dropped --/
structure EggCollection where
  gertrude : ℕ
  blanche : ℕ
  nancy : ℕ
  martha : ℕ
  dropped : ℕ

/-- Calculates the number of eggs Trevor has left --/
def eggsLeft (collection : EggCollection) : ℕ :=
  collection.gertrude + collection.blanche + collection.nancy + collection.martha - collection.dropped

/-- Theorem stating that Trevor has 9 eggs left --/
theorem trevor_eggs_left :
  ∃ (collection : EggCollection),
    collection.gertrude = 4 ∧
    collection.blanche = 3 ∧
    collection.nancy = 2 ∧
    collection.martha = 2 ∧
    collection.dropped = 2 ∧
    eggsLeft collection = 9 := by
  sorry

end NUMINAMATH_CALUDE_trevor_eggs_left_l858_85850


namespace NUMINAMATH_CALUDE_no_a_exists_for_union_range_of_a_for_intersection_l858_85810

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x = 0}

-- Define set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | a*x^2 - 2*x + 8 = 0}

-- Theorem 1: There does not exist a real number 'a' such that A ∪ B = {0, 2, 4}
theorem no_a_exists_for_union : ¬ ∃ a : ℝ, A ∪ B a = {0, 2, 4} := by
  sorry

-- Theorem 2: The range of 'a' when A ∩ B = B is {0} ∪ (1/8, +∞)
theorem range_of_a_for_intersection (a : ℝ) : 
  (A ∩ B a = B a) ↔ (a = 0 ∨ a > 1/8) := by
  sorry

end NUMINAMATH_CALUDE_no_a_exists_for_union_range_of_a_for_intersection_l858_85810


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l858_85848

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l858_85848


namespace NUMINAMATH_CALUDE_andrews_snacks_l858_85883

theorem andrews_snacks (num_friends : ℕ) (sandwiches_per_friend : ℕ) (cheese_crackers_per_friend : ℕ) (cookies_per_friend : ℕ) 
  (h1 : num_friends = 7)
  (h2 : sandwiches_per_friend = 5)
  (h3 : cheese_crackers_per_friend = 4)
  (h4 : cookies_per_friend = 3) :
  num_friends * sandwiches_per_friend + 
  num_friends * cheese_crackers_per_friend + 
  num_friends * cookies_per_friend = 84 := by
  sorry

end NUMINAMATH_CALUDE_andrews_snacks_l858_85883


namespace NUMINAMATH_CALUDE_range_of_b_l858_85839

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem range_of_b (f : ℝ → ℝ) (b : ℝ) :
  is_odd_function f →
  has_period f 4 →
  (∀ x ∈ Set.Ioo 0 2, f x = Real.log (x^2 - x + b)) →
  (∃ (zs : Finset ℝ), zs.card = 5 ∧ ∀ z ∈ zs, z ∈ Set.Icc (-2) 2 ∧ f z = 0) →
  b ∈ Set.Ioo (1/4) 1 ∪ {5/4} :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l858_85839


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l858_85892

/-- A geometric sequence is a sequence where the ratio between successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Nat.factorial n)

theorem geometric_sequence_first_term :
  ∀ a : ℕ → ℝ,
  IsGeometricSequence a →
  a 4 = factorial 6 →
  a 6 = factorial 7 →
  a 1 = (720 : ℝ) * Real.sqrt 7 / 49 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l858_85892


namespace NUMINAMATH_CALUDE_pencil_count_l858_85842

/-- Given the ratio of pens to pencils and their difference, calculate the number of pencils -/
theorem pencil_count (x : ℕ) (h1 : 6 * x = 5 * x + 6) : 6 * x = 36 := by
  sorry

#check pencil_count

end NUMINAMATH_CALUDE_pencil_count_l858_85842


namespace NUMINAMATH_CALUDE_fourth_square_area_l858_85849

-- Define the triangles and their properties
def triangle_PQR (PQ PR QR : ℝ) : Prop :=
  PQ^2 + PR^2 = QR^2 ∧ PQ = 5 ∧ PR = 7

def triangle_PRS (PR PS RS : ℝ) : Prop :=
  PR^2 + PS^2 = RS^2 ∧ PS = 8 ∧ PR = 7

-- Theorem statement
theorem fourth_square_area 
  (PQ PR QR PS RS : ℝ) 
  (h1 : triangle_PQR PQ PR QR) 
  (h2 : triangle_PRS PR PS RS) : 
  RS^2 = 113 := by
sorry

end NUMINAMATH_CALUDE_fourth_square_area_l858_85849


namespace NUMINAMATH_CALUDE_min_transactions_to_identify_coins_l858_85812

/-- Represents the set of coin values available -/
def CoinValues : Finset Nat := {1, 2, 5, 10, 20}

/-- The cost of one candy in florins -/
def CandyCost : Nat := 1

/-- Represents a vending machine transaction -/
structure Transaction where
  coin_inserted : Nat
  change_returned : Nat

/-- Function to determine if all coin values can be identified -/
def can_identify_all_coins (transactions : List Transaction) : Prop :=
  ∀ c ∈ CoinValues, ∃ t ∈ transactions, t.coin_inserted = c ∨ t.change_returned = c - CandyCost

/-- The main theorem stating that 4 is the minimum number of transactions required -/
theorem min_transactions_to_identify_coins :
  (∃ transactions : List Transaction, transactions.length = 4 ∧ can_identify_all_coins transactions) ∧
  (∀ transactions : List Transaction, transactions.length < 4 → ¬ can_identify_all_coins transactions) :=
sorry

end NUMINAMATH_CALUDE_min_transactions_to_identify_coins_l858_85812


namespace NUMINAMATH_CALUDE_tank_dimension_l858_85860

theorem tank_dimension (x : ℝ) : 
  x > 0 ∧ 
  (2 * (x * 5 + x * 2 + 5 * 2)) * 20 = 1240 → 
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_tank_dimension_l858_85860


namespace NUMINAMATH_CALUDE_vector_expression_l858_85861

theorem vector_expression (a b c : ℝ × ℝ) :
  a = (1, 2) →
  a + b = (0, 3) →
  c = (1, 5) →
  c = 2 • a + b := by
sorry

end NUMINAMATH_CALUDE_vector_expression_l858_85861


namespace NUMINAMATH_CALUDE_simplest_form_sqrt_l858_85826

/-- A number is a perfect square if it's the product of an integer with itself -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A square root is in simplest form if it cannot be simplified further -/
def is_simplest_form (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, n = a * b ∧ is_perfect_square a ∧ b > 1)

/-- The square root of a fraction is in simplest form if it cannot be simplified further -/
def is_simplest_form_frac (n d : ℕ) : Prop :=
  ¬(∃ a b c : ℕ, n = a * b ∧ d = a * c ∧ is_perfect_square a ∧ (b > 1 ∨ c > 1))

theorem simplest_form_sqrt :
  is_simplest_form 14 ∧
  ¬is_simplest_form 12 ∧
  ¬is_simplest_form 8 ∧
  ¬is_simplest_form_frac 1 3 :=
sorry

end NUMINAMATH_CALUDE_simplest_form_sqrt_l858_85826


namespace NUMINAMATH_CALUDE_polynomial_division_result_l858_85888

variables {a p x : ℝ}

theorem polynomial_division_result :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l858_85888


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l858_85813

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l858_85813


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l858_85869

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | |x - 1| < 1}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l858_85869


namespace NUMINAMATH_CALUDE_largest_perimeter_l858_85823

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  h1 : side1 = 7
  h2 : side2 = 9
  h3 : side3 % 3 = 0

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of the triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter --/
theorem largest_perimeter :
  ∀ t : Triangle, is_valid_triangle t →
  ∃ max_t : Triangle, is_valid_triangle max_t ∧
  perimeter max_t = 31 ∧
  ∀ other_t : Triangle, is_valid_triangle other_t →
  perimeter other_t ≤ perimeter max_t :=
sorry

end NUMINAMATH_CALUDE_largest_perimeter_l858_85823


namespace NUMINAMATH_CALUDE_simplify_fraction_l858_85871

theorem simplify_fraction : (333 : ℚ) / 9999 * 99 = 37 / 101 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l858_85871


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_is_integer_l858_85866

theorem sqrt_sum_squares_is_integer : ∃ (z : ℕ), z * z = 25530 * 25530 + 29464 * 29464 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_is_integer_l858_85866


namespace NUMINAMATH_CALUDE_select_with_abc_must_select_with_one_abc_select_with_at_most_two_abc_l858_85882

-- Define the total number of people
def total_people : ℕ := 12

-- Define the number of people to be selected
def select_count : ℕ := 5

-- Define the number of special people (A, B, C)
def special_people : ℕ := 3

-- Theorem 1: When A, B, and C must be chosen
theorem select_with_abc_must (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose (n - s) (k - s) = 36 :=
sorry

-- Theorem 2: When only one among A, B, and C is chosen
theorem select_with_one_abc (n : ℕ) (k : ℕ) (s : ℕ) :
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose s 1 * Nat.choose (n - s) (k - 1) = 378 :=
sorry

-- Theorem 3: When at most two among A, B, and C are chosen
theorem select_with_at_most_two_abc (n : ℕ) (k : ℕ) (s : ℕ) :
  n = total_people ∧ k = select_count ∧ s = special_people →
  Nat.choose n k - Nat.choose (n - s) (k - s) = 756 :=
sorry

end NUMINAMATH_CALUDE_select_with_abc_must_select_with_one_abc_select_with_at_most_two_abc_l858_85882


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l858_85890

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l858_85890


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l858_85878

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l858_85878


namespace NUMINAMATH_CALUDE_sum_of_integers_l858_85835

theorem sum_of_integers (x y : ℤ) : 
  3 * x + 2 * y = 115 → (x = 25 ∨ y = 25) → (x = 20 ∨ y = 20) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l858_85835


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l858_85809

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (2/5 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (2/5 : ℂ) - (1/3 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l858_85809


namespace NUMINAMATH_CALUDE_yanna_apples_kept_l858_85845

def apples_kept (initial : ℕ) (given_to_zenny : ℕ) (given_to_andrea : ℕ) : ℕ :=
  initial - given_to_zenny - given_to_andrea

theorem yanna_apples_kept :
  apples_kept 60 18 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_yanna_apples_kept_l858_85845


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l858_85877

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l858_85877


namespace NUMINAMATH_CALUDE_find_incorrect_value_l858_85880

/-- Represents the problem of finding the incorrect value in a mean calculation --/
theorem find_incorrect_value (n : ℕ) (initial_mean correct_mean correct_value : ℚ) :
  n = 30 ∧ 
  initial_mean = 180 ∧ 
  correct_mean = 180 + 2/3 ∧ 
  correct_value = 155 →
  ∃ incorrect_value : ℚ,
    incorrect_value = 175 ∧
    n * initial_mean = (n - 1) * correct_mean + incorrect_value ∧
    n * correct_mean = (n - 1) * correct_mean + correct_value :=
by sorry

end NUMINAMATH_CALUDE_find_incorrect_value_l858_85880


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l858_85828

/-- Represents a normal distribution with mean μ and standard deviation σ -/
noncomputable def NormalDistribution (μ σ : ℝ) : Type :=
  ℝ → ℝ

/-- The probability that a random variable X from a normal distribution
    falls within the interval [a, b] -/
noncomputable def prob_between (X : NormalDistribution μ σ) (a b : ℝ) : ℝ :=
  sorry

/-- The probability that a random variable X from a normal distribution
    is greater than or equal to a given value -/
noncomputable def prob_ge (X : NormalDistribution μ σ) (a : ℝ) : ℝ :=
  sorry

theorem normal_distribution_symmetry 
  (X : NormalDistribution 100 σ) 
  (h : prob_between X 80 120 = 3/4) : 
  prob_ge X 120 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l858_85828


namespace NUMINAMATH_CALUDE_equal_cake_distribution_l858_85854

theorem equal_cake_distribution (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) : 
  total_cakes = 18 → 
  num_children = 3 → 
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
sorry

end NUMINAMATH_CALUDE_equal_cake_distribution_l858_85854


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l858_85829

theorem quadratic_roots_property (m : ℝ) (hm : m ≠ -1) :
  let f : ℝ → ℝ := λ x => (m + 1) * x^2 + 4 * m * x + m - 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ (x₁ < -1 ∨ x₂ < -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l858_85829


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l858_85820

/-- 
Given three terms in the form (10 + x), (40 + x), and (90 + x),
prove that x = 35 is the unique solution for which these terms form a geometric progression.
-/
theorem geometric_progression_solution : 
  ∃! x : ℝ, (∃ r : ℝ, r ≠ 0 ∧ (40 + x) = (10 + x) * r ∧ (90 + x) = (40 + x) * r) ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l858_85820


namespace NUMINAMATH_CALUDE_expression_equality_l858_85843

theorem expression_equality : 
  (1/2)⁻¹ + 4 * Real.cos (60 * π / 180) - |-3| + Real.sqrt 9 - (-2023)^0 + (-1)^(2023-1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l858_85843


namespace NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l858_85862

def alice_number : ℕ := 72

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_number_with_same_prime_factors :
  ∃ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_same_prime_factors_l858_85862


namespace NUMINAMATH_CALUDE_lemonade_problem_l858_85885

theorem lemonade_problem (x : ℝ) :
  x > 0 ∧
  (x + (x / 8 + 2) = (3 / 2 * x) - (x / 8 + 2)) →
  x + (3 / 2 * x) = 40 := by
sorry

end NUMINAMATH_CALUDE_lemonade_problem_l858_85885


namespace NUMINAMATH_CALUDE_andy_work_hours_l858_85841

-- Define the variables and constants
def hourly_rate : ℝ := 9
def restring_fee : ℝ := 15
def grommet_fee : ℝ := 10
def stencil_fee : ℝ := 1
def total_earnings : ℝ := 202
def racquets_strung : ℕ := 7
def grommets_changed : ℕ := 2
def stencils_painted : ℕ := 5

-- State the theorem
theorem andy_work_hours :
  ∃ (hours : ℝ),
    hours * hourly_rate +
    racquets_strung * restring_fee +
    grommets_changed * grommet_fee +
    stencils_painted * stencil_fee = total_earnings ∧
    hours = 8 := by sorry

end NUMINAMATH_CALUDE_andy_work_hours_l858_85841


namespace NUMINAMATH_CALUDE_curve_self_intersection_l858_85838

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2 - 3, t^3 - 6*t + 2)

-- Theorem statement
theorem curve_self_intersection :
  ∃! p : ℝ × ℝ, ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ curve t₁ = p ∧ curve t₂ = p ∧ p = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l858_85838


namespace NUMINAMATH_CALUDE_ninth_grade_test_attendance_l858_85832

theorem ninth_grade_test_attendance :
  let total_students : ℕ := 180
  let bombed_finals : ℕ := total_students / 4
  let remaining_students : ℕ := total_students - bombed_finals
  let passed_finals : ℕ := 70
  let less_than_d : ℕ := 20
  let took_test : ℕ := passed_finals + less_than_d
  let didnt_show_up : ℕ := remaining_students - took_test
  (didnt_show_up : ℚ) / remaining_students = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_test_attendance_l858_85832


namespace NUMINAMATH_CALUDE_sam_age_l858_85815

theorem sam_age (drew_age : ℕ) (sam_age : ℕ) : 
  drew_age + sam_age = 54 →
  sam_age = drew_age / 2 →
  sam_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sam_age_l858_85815


namespace NUMINAMATH_CALUDE_cube_in_pyramid_volume_l858_85881

/-- A pyramid with a square base and isosceles right triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (is_square_base : base_side > 0)
  (lateral_faces_isosceles_right : True)

/-- A cube placed inside a pyramid -/
structure CubeInPyramid :=
  (pyramid : Pyramid)
  (bottom_on_base : True)
  (top_touches_midpoints : True)

/-- The volume of a cube -/
def cube_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

/-- Theorem: The volume of the cube in the given pyramid configuration is 1 -/
theorem cube_in_pyramid_volume 
  (p : Pyramid) 
  (c : CubeInPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : c.pyramid = p) : 
  ∃ (edge_length : ℝ), cube_volume edge_length = 1 :=
sorry

end NUMINAMATH_CALUDE_cube_in_pyramid_volume_l858_85881


namespace NUMINAMATH_CALUDE_journey_distance_on_foot_l858_85857

theorem journey_distance_on_foot 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_on_foot : ℝ) 
  (speed_on_bicycle : ℝ) 
  (h1 : total_distance = 80) 
  (h2 : total_time = 7) 
  (h3 : speed_on_foot = 8) 
  (h4 : speed_on_bicycle = 16) :
  ∃ (distance_on_foot : ℝ),
    distance_on_foot = 32 ∧
    ∃ (distance_on_bicycle : ℝ),
      distance_on_foot + distance_on_bicycle = total_distance ∧
      distance_on_foot / speed_on_foot + distance_on_bicycle / speed_on_bicycle = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_on_foot_l858_85857


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l858_85831

def is_midpoint (m x y : ℝ × ℝ) : Prop :=
  m.1 = (x.1 + y.1) / 2 ∧ m.2 = (x.2 + y.2) / 2

theorem line_segment_endpoint (endpoint1 midpoint : ℝ × ℝ) 
  (h : is_midpoint midpoint endpoint1 (1, 18)) : 
  endpoint1 = (5, 2) ∧ midpoint = (3, 10) → (1, 18) = (1, 18) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l858_85831


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l858_85805

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / 9 - y^2 / a = 1

-- Define the right focus
def right_focus (a : ℝ) : Prop := hyperbola (Real.sqrt 13) 0 a

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

-- Theorem statement
theorem hyperbola_asymptotes (a : ℝ) :
  right_focus a → ∀ x y, hyperbola x y a → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l858_85805


namespace NUMINAMATH_CALUDE_spinner_probability_l858_85859

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → p_A + p_B + p_C + p_D = 1 → p_D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l858_85859


namespace NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l858_85893

theorem sum_of_squares_in_ratio (x y z : ℝ) : 
  x + y + z = 9 ∧ y = 2*x ∧ z = 4*x → x^2 + y^2 + z^2 = 1701 / 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l858_85893


namespace NUMINAMATH_CALUDE_parabola_properties_l858_85891

-- Define the parabola C
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def Directrix (x : ℝ) : Prop := x = -1

-- Define a point on the parabola
def PointOnParabola (p : ℝ × ℝ) : Prop := Parabola p.1 p.2

-- Define a line passing through two points
def Line (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

-- Theorem statement
theorem parabola_properties :
  ∀ (M N : ℝ × ℝ),
  Directrix M.1 ∧ Directrix N.1 →
  M.2 * N.2 = -4 →
  ∃ (A B : ℝ × ℝ) (F : ℝ × ℝ → ℝ × ℝ),
    PointOnParabola A ∧ PointOnParabola B ∧
    Line (0, 0) M A.1 A.2 ∧
    Line (0, 0) N B.1 B.2 ∧
    (∀ (x y : ℝ), Line A B x y → Line A B (F A).1 (F A).2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l858_85891


namespace NUMINAMATH_CALUDE_range_of_p_l858_85894

-- Define the sequence a_n
def a (n : ℕ+) : ℝ := (-1 : ℝ)^(n.val - 1) * (2 * n.val - 1)

-- Define the sum S_n
def S (n : ℕ+) : ℝ := (-1 : ℝ)^(n.val - 1) * n.val

-- Theorem statement
theorem range_of_p (p : ℝ) :
  (∀ n : ℕ+, (a (n + 1) - p) * (a n - p) < 0) ↔ -3 < p ∧ p < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l858_85894


namespace NUMINAMATH_CALUDE_max_salary_baseball_team_l858_85895

/-- Represents the maximum salary for a single player in a baseball team under given constraints -/
def max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_budget : ℕ) : ℕ :=
  total_budget - (num_players - 1) * min_salary

/-- Theorem stating the maximum possible salary for a single player under given constraints -/
theorem max_salary_baseball_team :
  max_player_salary 18 20000 600000 = 260000 :=
by sorry

end NUMINAMATH_CALUDE_max_salary_baseball_team_l858_85895


namespace NUMINAMATH_CALUDE_simplify_expressions_l858_85819

variable (a b : ℝ)

theorem simplify_expressions :
  (-2 * a * b - a^2 + 3 * a * b - 5 * a^2 = a * b - 6 * a^2) ∧
  ((4 * a * b - b^2) - 2 * (a^2 + 2 * a * b - b^2) = b^2 - 2 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l858_85819


namespace NUMINAMATH_CALUDE_total_notes_count_l858_85886

def total_amount : ℕ := 10350
def note_50_value : ℕ := 50
def note_500_value : ℕ := 500
def num_50_notes : ℕ := 17

theorem total_notes_count : 
  ∃ (num_500_notes : ℕ), 
    num_50_notes * note_50_value + num_500_notes * note_500_value = total_amount ∧
    num_50_notes + num_500_notes = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_notes_count_l858_85886


namespace NUMINAMATH_CALUDE_sin_15_degrees_l858_85872

theorem sin_15_degrees : Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_degrees_l858_85872


namespace NUMINAMATH_CALUDE_average_price_per_book_l858_85853

theorem average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) 
  (h1 : books1 = 55) (h2 : price1 = 1500) (h3 : books2 = 60) (h4 : price2 = 340) :
  (price1 + price2) / (books1 + books2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l858_85853


namespace NUMINAMATH_CALUDE_julia_watch_collection_l858_85824

theorem julia_watch_collection (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = (silver_watches + bronze_watches + gold_watches) / 10 →
  silver_watches + bronze_watches + gold_watches = 88 :=
by sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l858_85824


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l858_85840

theorem no_solutions_for_equation :
  ∀ (x y : ℝ), x^2 + y^2 - 2*y + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l858_85840


namespace NUMINAMATH_CALUDE_rachel_homework_l858_85821

/-- Rachel's homework problem -/
theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 5 → reading_pages = 2 → math_pages + reading_pages = 7 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l858_85821


namespace NUMINAMATH_CALUDE_lisa_walking_speed_l858_85808

/-- The number of meters Lisa walks per minute -/
def meters_per_minute (total_distance : ℕ) (days : ℕ) (hours_per_day : ℕ) (minutes_per_hour : ℕ) : ℚ :=
  (total_distance : ℚ) / (days * hours_per_day * minutes_per_hour)

/-- Proof that Lisa walks 10 meters per minute -/
theorem lisa_walking_speed :
  let total_distance := 1200
  let days := 2
  let hours_per_day := 1
  let minutes_per_hour := 60
  meters_per_minute total_distance days hours_per_day minutes_per_hour = 10 := by
  sorry

#eval meters_per_minute 1200 2 1 60

end NUMINAMATH_CALUDE_lisa_walking_speed_l858_85808


namespace NUMINAMATH_CALUDE_lentil_dishes_count_l858_85897

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_lentils : ℕ)
  (beans_seitan : ℕ)
  (tempeh_lentils : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)
  (only_tempeh : ℕ)

/-- The conditions of the vegan restaurant menu -/
def menu_conditions (m : VeganMenu) : Prop :=
  m.total_dishes = 20 ∧
  m.beans_lentils = 3 ∧
  m.beans_seitan = 4 ∧
  m.tempeh_lentils = 2 ∧
  m.only_beans = 2 * m.only_tempeh ∧
  m.only_seitan = 3 * m.only_tempeh ∧
  m.total_dishes = m.beans_lentils + m.beans_seitan + m.tempeh_lentils +
                   m.only_beans + m.only_seitan + m.only_lentils + m.only_tempeh

/-- The theorem stating that the number of dishes with lentils is 10 -/
theorem lentil_dishes_count (m : VeganMenu) :
  menu_conditions m → m.only_lentils + m.beans_lentils + m.tempeh_lentils = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_lentil_dishes_count_l858_85897


namespace NUMINAMATH_CALUDE_simple_interest_years_l858_85879

/-- Calculates the number of years for which a sum was put at simple interest, given the principal amount and the additional interest earned with a 1% rate increase. -/
def calculateYears (principal : ℚ) (additionalInterest : ℚ) : ℚ :=
  (100 * additionalInterest) / principal

theorem simple_interest_years :
  let principal : ℚ := 2300
  let additionalInterest : ℚ := 69
  calculateYears principal additionalInterest = 3 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_years_l858_85879


namespace NUMINAMATH_CALUDE_gcf_of_48_180_98_l858_85887

theorem gcf_of_48_180_98 : Nat.gcd 48 (Nat.gcd 180 98) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_48_180_98_l858_85887


namespace NUMINAMATH_CALUDE_max_triangle_area_l858_85898

theorem max_triangle_area (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (htri : a + b > c ∧ a + c > b ∧ b + c > a) :
  ∃ (area : ℝ), area ≤ 1 ∧ ∀ (other_area : ℝ), 
    (∃ (x y z : ℝ), 0 < x ∧ x ≤ 1 ∧ 1 ≤ y ∧ y ≤ 2 ∧ 2 ≤ z ∧ z ≤ 3 ∧ 
      x + y > z ∧ x + z > y ∧ y + z > x ∧
      other_area = (x + y + z) * (- x + y + z) * (x - y + z) * (x + y - z) / (4 * (x + y + z))) →
    other_area ≤ area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l858_85898


namespace NUMINAMATH_CALUDE_remainder_7531_mod_11_l858_85811

def digit_sum (n : ℕ) : ℕ := sorry

theorem remainder_7531_mod_11 :
  ∃ k : ℤ, 7531 = 11 * k + 5 :=
by
  have h1 : ∀ n : ℕ, ∃ k : ℤ, n = 11 * k + (digit_sum n % 11) := sorry
  sorry

end NUMINAMATH_CALUDE_remainder_7531_mod_11_l858_85811


namespace NUMINAMATH_CALUDE_quadratic_factorization_l858_85837

theorem quadratic_factorization (x : ℝ) : x^2 - x - 42 = (x + 6) * (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l858_85837


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l858_85847

theorem tic_tac_toe_tie_probability (john_win_prob martha_win_prob : ℚ) 
  (h1 : john_win_prob = 4 / 9)
  (h2 : martha_win_prob = 5 / 12) :
  1 - (john_win_prob + martha_win_prob) = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l858_85847


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l858_85873

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l858_85873


namespace NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l858_85855

/-- A polygon in the figure --/
inductive Polygon
| EquilateralTriangle
| Square
| RegularHexagon

/-- The figure consisting of multiple polygons --/
structure Figure where
  polygons : List Polygon
  triangleSideLength : ℝ
  squareSideLength : ℝ
  hexagonSideLength : ℝ

/-- The polyhedron formed by folding the figure --/
structure Polyhedron where
  figure : Figure

/-- Calculate the volume of the polyhedron --/
def calculateVolume (p : Polyhedron) : ℝ :=
  sorry

/-- The theorem stating that the volume of the specific polyhedron is 8 --/
theorem volume_of_specific_polyhedron :
  let fig : Figure := {
    polygons := [Polygon.EquilateralTriangle, Polygon.EquilateralTriangle, Polygon.EquilateralTriangle,
                 Polygon.Square, Polygon.Square, Polygon.Square,
                 Polygon.RegularHexagon],
    triangleSideLength := 2,
    squareSideLength := 2,
    hexagonSideLength := 1
  }
  let poly : Polyhedron := { figure := fig }
  calculateVolume poly = 8 :=
sorry

end NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l858_85855


namespace NUMINAMATH_CALUDE_luncheon_tables_l858_85867

theorem luncheon_tables (invited : ℕ) (no_show : ℕ) (per_table : ℕ) 
  (h1 : invited = 18) 
  (h2 : no_show = 12) 
  (h3 : per_table = 3) : 
  (invited - no_show) / per_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_tables_l858_85867


namespace NUMINAMATH_CALUDE_computer_contract_probability_l858_85818

theorem computer_contract_probability (p_not_software : ℝ) (p_at_least_one : ℝ) (p_both : ℝ) 
  (h1 : p_not_software = 3/5)
  (h2 : p_at_least_one = 5/6)
  (h3 : p_both = 0.31666666666666654) :
  let p_software := 1 - p_not_software
  let p_hardware := p_at_least_one + p_both - p_software
  p_hardware = 0.75 := by
sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l858_85818


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l858_85899

theorem contrapositive_real_roots :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔
  (∀ m : ℝ, (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l858_85899


namespace NUMINAMATH_CALUDE_book_pages_proof_l858_85803

/-- The number of pages Jack reads per day -/
def pages_per_day : ℕ := 23

/-- The number of pages Jack reads on the last day -/
def last_day_pages : ℕ := 9

/-- The total number of pages in the book -/
def total_pages : ℕ := 32

theorem book_pages_proof :
  ∃ (full_days : ℕ), total_pages = pages_per_day * full_days + last_day_pages :=
by sorry

end NUMINAMATH_CALUDE_book_pages_proof_l858_85803


namespace NUMINAMATH_CALUDE_mrs_snyder_income_l858_85827

/-- Mrs. Snyder's previous monthly income --/
def previous_income : ℝ := 1000

/-- Mrs. Snyder's salary increase --/
def salary_increase : ℝ := 600

/-- Percentage of previous income spent on rent and utilities --/
def previous_percentage : ℝ := 0.40

/-- Percentage of new income spent on rent and utilities --/
def new_percentage : ℝ := 0.25

theorem mrs_snyder_income :
  previous_income * previous_percentage = 
  (previous_income + salary_increase) * new_percentage := by
  sorry

#check mrs_snyder_income

end NUMINAMATH_CALUDE_mrs_snyder_income_l858_85827


namespace NUMINAMATH_CALUDE_exponent_of_p_in_product_l858_85833

theorem exponent_of_p_in_product (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  ∃ (a b : ℕ), (a + 1) * (b + 1) = 32 ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_p_in_product_l858_85833


namespace NUMINAMATH_CALUDE_constant_altitude_triangle_l858_85800

/-- Given an equilateral triangle and a line through its center, prove the existence of a triangle
    with constant altitude --/
theorem constant_altitude_triangle (a : ℝ) (m : ℝ) :
  let A : ℝ × ℝ := (0, Real.sqrt 3 * a)
  let B : ℝ × ℝ := (-a, 0)
  let C : ℝ × ℝ := (a, 0)
  let O : ℝ × ℝ := (0, Real.sqrt 3 * a / 3)
  let N : ℝ × ℝ := (0, Real.sqrt 3 * a / 3)
  let M : ℝ × ℝ := (-Real.sqrt 3 * a / (3 * m), 0)
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let BN := Real.sqrt ((N.1 - B.1)^2 + (N.2 - B.2)^2)
  let MN := Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)
  ∃ (D E F : ℝ × ℝ),
    let h := Real.sqrt 6 * a / 3
    (E.1 - D.1)^2 + (E.2 - D.2)^2 = MN^2 ∧
    (F.1 - D.1)^2 + (F.2 - D.2)^2 = AM^2 ∧
    (F.1 - E.1)^2 + (F.2 - E.2)^2 = BN^2 ∧
    2 * (abs ((F.2 - E.2) * D.1 + (E.1 - F.1) * D.2 + (F.1 * E.2 - E.1 * F.2)) / Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)) = h :=
by
  sorry


end NUMINAMATH_CALUDE_constant_altitude_triangle_l858_85800


namespace NUMINAMATH_CALUDE_square_plus_square_l858_85846

theorem square_plus_square (x : ℝ) : x^2 + x^2 = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_square_l858_85846


namespace NUMINAMATH_CALUDE_value_of_M_l858_85856

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1500) ∧ (M = 2100) := by sorry

end NUMINAMATH_CALUDE_value_of_M_l858_85856
