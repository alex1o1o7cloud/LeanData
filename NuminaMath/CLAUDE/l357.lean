import Mathlib

namespace NUMINAMATH_CALUDE_number_of_children_l357_35790

theorem number_of_children (bottle_caps_per_child : ℕ) (total_bottle_caps : ℕ) : 
  bottle_caps_per_child = 5 → total_bottle_caps = 45 → 
  ∃ num_children : ℕ, num_children * bottle_caps_per_child = total_bottle_caps ∧ num_children = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l357_35790


namespace NUMINAMATH_CALUDE_triangle_with_120_degree_angle_l357_35703

theorem triangle_with_120_degree_angle : 
  ∃ (a b c : ℕ), 
    a = 3 ∧ b = 6 ∧ c = 7 ∧ 
    (a^2 + b^2 - c^2 : ℝ) / (2 * a * b : ℝ) = - (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_120_degree_angle_l357_35703


namespace NUMINAMATH_CALUDE_completing_square_l357_35779

theorem completing_square (x : ℝ) : x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l357_35779


namespace NUMINAMATH_CALUDE_direct_variation_with_constant_value_of_y_at_negative_ten_l357_35788

/-- A function representing direct variation with an additional constant. -/
def y (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem: If y(5) = 15 and c = 3, then y(-10) = -21 -/
theorem direct_variation_with_constant 
  (k : ℝ) 
  (h1 : y k 3 5 = 15) 
  : y k 3 (-10) = -21 := by
  sorry

/-- Corollary: The value of y when x = -10 and c = 3 is -21 -/
theorem value_of_y_at_negative_ten 
  (k : ℝ) 
  (h1 : y k 3 5 = 15) 
  : ∃ (y_value : ℝ), y k 3 (-10) = y_value ∧ y_value = -21 := by
  sorry

end NUMINAMATH_CALUDE_direct_variation_with_constant_value_of_y_at_negative_ten_l357_35788


namespace NUMINAMATH_CALUDE_number_problem_l357_35721

theorem number_problem : ∃ x : ℝ, x = 1/8 + 0.675 ∧ x = 0.800 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l357_35721


namespace NUMINAMATH_CALUDE_equation_solution_characterization_equation_unique_solution_characterization_l357_35702

/-- The equation has a solution -/
def has_solution (a b : ℝ) : Prop :=
  (a = 2 ∧ b = 3) ∨ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3)

/-- The solution is unique -/
def unique_solution (a b : ℝ) : Prop :=
  a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3

/-- The equation in question -/
def equation (x a b : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 3 ∧ (x - a) / (x - 2) + (x - b) / (x - 3) = 2

theorem equation_solution_characterization (a b : ℝ) :
  (∃ x, equation x a b) ↔ has_solution a b :=
sorry

theorem equation_unique_solution_characterization (a b : ℝ) :
  (∃! x, equation x a b) ↔ unique_solution a b :=
sorry

end NUMINAMATH_CALUDE_equation_solution_characterization_equation_unique_solution_characterization_l357_35702


namespace NUMINAMATH_CALUDE_zebra_fox_ratio_l357_35749

/-- Proves that the ratio of zebras to foxes is 3:1 given the conditions of the problem --/
theorem zebra_fox_ratio :
  let total_animals : ℕ := 100
  let num_cows : ℕ := 20
  let num_foxes : ℕ := 15
  let num_sheep : ℕ := 20
  let num_zebras : ℕ := total_animals - (num_cows + num_foxes + num_sheep)
  (num_zebras : ℚ) / num_foxes = 3 := by sorry

end NUMINAMATH_CALUDE_zebra_fox_ratio_l357_35749


namespace NUMINAMATH_CALUDE_concentric_circles_radii_l357_35762

theorem concentric_circles_radii 
  (r R : ℝ) 
  (h_positive : 0 < r ∧ 0 < R) 
  (h_order : r < R) 
  (h_min_distance : R - r = 2) 
  (h_max_distance : R + r = 16) : 
  r = 7 ∧ R = 9 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_l357_35762


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l357_35711

theorem diophantine_equation_solution (m n x y : ℕ) : m ≥ 2 ∧ n ≥ 2 ∧ x^n + y^n = 3^m →
  ((x = 2 ∧ y = 1 ∧ n = 3 ∧ m = 2) ∨ (x = 1 ∧ y = 2 ∧ n = 3 ∧ m = 2)) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l357_35711


namespace NUMINAMATH_CALUDE_smallest_student_count_l357_35705

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  7 * counts.tenth = 4 * counts.ninth ∧
  21 * counts.eleventh = 10 * counts.ninth

/-- The theorem stating the smallest possible total number of students --/
theorem smallest_student_count : 
  ∃ (counts : StudentCounts), 
    satisfiesRatios counts ∧ 
    counts.ninth + counts.tenth + counts.eleventh = 43 ∧
    (∀ (other : StudentCounts), 
      satisfiesRatios other → 
      other.ninth + other.tenth + other.eleventh ≥ 43) :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_count_l357_35705


namespace NUMINAMATH_CALUDE_hyperbola_equation_l357_35732

-- Define the hyperbola
def Hyperbola (x y : ℝ) : Prop := y^2 - x^2/2 = 1

-- State the theorem
theorem hyperbola_equation :
  ∀ (c a : ℝ),
  c = Real.sqrt 3 →
  a = Real.sqrt 3 - 1 →
  ∀ (x y : ℝ),
  Hyperbola x y ↔ 
  (x^2 / (c^2 - a^2) - y^2 / c^2 = 1) ∧
  (c^2 - a^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l357_35732


namespace NUMINAMATH_CALUDE_initial_pens_l357_35794

theorem initial_pens (P : ℕ) : 2 * (P + 22) - 19 = 65 ↔ P = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_pens_l357_35794


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l357_35752

-- Define the system of equations
def equation1 (x : Fin 8 → ℝ) : Prop := x 0 + x 1 + x 2 = 6
def equation2 (x : Fin 8 → ℝ) : Prop := x 1 + x 2 + x 3 = 9
def equation3 (x : Fin 8 → ℝ) : Prop := x 2 + x 3 + x 4 = 3
def equation4 (x : Fin 8 → ℝ) : Prop := x 3 + x 4 + x 5 = -3
def equation5 (x : Fin 8 → ℝ) : Prop := x 4 + x 5 + x 6 = -9
def equation6 (x : Fin 8 → ℝ) : Prop := x 5 + x 6 + x 7 = -6
def equation7 (x : Fin 8 → ℝ) : Prop := x 6 + x 7 + x 0 = -2
def equation8 (x : Fin 8 → ℝ) : Prop := x 7 + x 0 + x 1 = 2

-- Define the solution
def solution : Fin 8 → ℝ
| 0 => 1
| 1 => 2
| 2 => 3
| 3 => 4
| 4 => -4
| 5 => -3
| 6 => -2
| 7 => -1

-- Theorem statement
theorem solution_satisfies_equations :
  equation1 solution ∧
  equation2 solution ∧
  equation3 solution ∧
  equation4 solution ∧
  equation5 solution ∧
  equation6 solution ∧
  equation7 solution ∧
  equation8 solution :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l357_35752


namespace NUMINAMATH_CALUDE_ellipse_circle_area_relation_sum_of_x_values_l357_35745

theorem ellipse_circle_area_relation (x : ℝ) : 
  let circle_radius := x - 2
  let ellipse_semi_major := x - 3
  let ellipse_semi_minor := x + 4
  π * ellipse_semi_major * ellipse_semi_minor = 2 * π * circle_radius^2 →
  x = 4 ∨ x = 5 :=
by
  sorry

theorem sum_of_x_values : 
  ∃ (x₁ x₂ : ℝ), 
    (let circle_radius := x₁ - 2
     let ellipse_semi_major := x₁ - 3
     let ellipse_semi_minor := x₁ + 4
     π * ellipse_semi_major * ellipse_semi_minor = 2 * π * circle_radius^2) ∧
    (let circle_radius := x₂ - 2
     let ellipse_semi_major := x₂ - 3
     let ellipse_semi_minor := x₂ + 4
     π * ellipse_semi_major * ellipse_semi_minor = 2 * π * circle_radius^2) ∧
    x₁ + x₂ = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_circle_area_relation_sum_of_x_values_l357_35745


namespace NUMINAMATH_CALUDE_complex_absolute_value_l357_35740

theorem complex_absolute_value (z : ℂ) (h : z = 1 + Complex.I) : 
  Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l357_35740


namespace NUMINAMATH_CALUDE_probability_at_most_five_digits_value_l357_35774

/-- The probability of having at most five different digits in a randomly chosen
    sequence of seven digits, where each digit can be any of the digits 0 through 9. -/
def probability_at_most_five_digits : ℚ :=
  1 - (↑(Nat.choose 10 6 * 6 * (7 * 6 * 5 * 4 * 3 * 2 * 1 / 2)) / 10^7 +
       ↑(Nat.choose 10 7 * (7 * 6 * 5 * 4 * 3 * 2 * 1)) / 10^7)

/-- Theorem stating that the probability of having at most five different digits
    in the described sequence is equal to 0.622. -/
theorem probability_at_most_five_digits_value :
  probability_at_most_five_digits = 311 / 500 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_most_five_digits_value_l357_35774


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l357_35742

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) (1/2 : ℝ) ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (1/2 : ℝ) → f x ≤ f c ∧
  f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l357_35742


namespace NUMINAMATH_CALUDE_page_number_added_twice_l357_35771

theorem page_number_added_twice (n : ℕ) (h : ∃ k : ℕ, k ≤ n ∧ (n * (n + 1)) / 2 + k = 2900) : 
  ∃ k : ℕ, k ≤ n ∧ (n * (n + 1)) / 2 + k = 2900 ∧ k = 50 := by
  sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l357_35771


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l357_35701

def p (x : ℝ) : ℝ := x^3 - 8*x^2 + 19*x - 12

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 3 ∨ x = 4) ∧
  p 1 = 0 ∧ p 3 = 0 ∧ p 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l357_35701


namespace NUMINAMATH_CALUDE_erik_bread_loaves_l357_35768

/-- Calculates the number of bread loaves bought given the initial amount,
    number of orange juice cartons, costs, and remaining amount. -/
def breadLoavesBought (initialAmount : ℕ) (orangeJuiceCartons : ℕ) 
                      (breadCost : ℕ) (orangeJuiceCost : ℕ) 
                      (remainingAmount : ℕ) : ℕ :=
  (initialAmount - remainingAmount - orangeJuiceCartons * orangeJuiceCost) / breadCost

/-- Theorem stating that Erik bought 3 loaves of bread -/
theorem erik_bread_loaves : 
  breadLoavesBought 86 3 3 6 59 = 3 := by
  sorry

#eval breadLoavesBought 86 3 3 6 59

end NUMINAMATH_CALUDE_erik_bread_loaves_l357_35768


namespace NUMINAMATH_CALUDE_number_of_mappings_l357_35789

theorem number_of_mappings (n m : ℕ) : 
  (Finset.univ : Finset (Fin n → Fin m)).card = m ^ n := by
  sorry

end NUMINAMATH_CALUDE_number_of_mappings_l357_35789


namespace NUMINAMATH_CALUDE_bank_account_balances_l357_35728

/-- Calculates the final balances of two bank accounts after a series of transactions --/
theorem bank_account_balances
  (primary_initial : ℝ)
  (secondary_initial : ℝ)
  (primary_deposit : ℝ)
  (secondary_deposit : ℝ)
  (primary_spend : ℝ)
  (save_percentage : ℝ)
  (h1 : primary_initial = 3179.37)
  (h2 : secondary_initial = 1254.12)
  (h3 : primary_deposit = 21.85)
  (h4 : secondary_deposit = 150)
  (h5 : primary_spend = 87.41)
  (h6 : save_percentage = 0.15)
  : ∃ (primary_available secondary_final : ℝ),
    primary_available = 2646.74 ∧
    secondary_final = 1404.12 := by
  sorry


end NUMINAMATH_CALUDE_bank_account_balances_l357_35728


namespace NUMINAMATH_CALUDE_jared_current_age_l357_35753

/-- Represents a person's age at different points in time -/
structure PersonAge where
  current : ℕ
  twoYearsAgo : ℕ
  fiveYearsLater : ℕ

/-- The problem statement -/
theorem jared_current_age (tom : PersonAge) (jared : PersonAge) : 
  tom.fiveYearsLater = 30 →
  jared.twoYearsAgo = 2 * tom.twoYearsAgo →
  jared.current = 48 := by
  sorry

end NUMINAMATH_CALUDE_jared_current_age_l357_35753


namespace NUMINAMATH_CALUDE_number_ratio_l357_35797

theorem number_ratio (A B C : ℝ) 
  (h1 : (A + B + C) / (A + B - C) = 4/3)
  (h2 : (A + B) / (B + C) = 7/6) :
  ∃ (k : ℝ), k ≠ 0 ∧ A = 2*k ∧ B = 5*k ∧ C = k := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l357_35797


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l357_35769

theorem binomial_coefficient_problem (m : ℕ) : 
  (1 : ℚ) / (Nat.choose 5 m) - (1 : ℚ) / (Nat.choose 6 m) = 7 / (10 * (Nat.choose 7 m)) → 
  Nat.choose 8 m = 28 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l357_35769


namespace NUMINAMATH_CALUDE_max_area_circular_sector_l357_35793

/-- Given a circular sector with perimeter 2p, prove that the maximum area is p^2/4 -/
theorem max_area_circular_sector (p : ℝ) (hp : p > 0) :
  ∃ (R : ℝ), R > 0 ∧
    (∀ (S : ℝ → ℝ), (∀ r, r > 0 → S r = r * (p - r)) →
      (∀ r, r > 0 → S r ≤ S R) ∧ S R = p^2 / 4) :=
sorry

end NUMINAMATH_CALUDE_max_area_circular_sector_l357_35793


namespace NUMINAMATH_CALUDE_double_points_imply_m_less_than_one_l357_35747

/-- A double point is a point where the ordinate is twice its abscissa -/
def DoublePoint (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m

theorem double_points_imply_m_less_than_one (m : ℝ) 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : DoublePoint x₁ y₁)
  (h₂ : DoublePoint x₂ y₂)
  (h₃ : f m x₁ = y₁)
  (h₄ : f m x₂ = y₂)
  (h₅ : x₁ < 1)
  (h₆ : 1 < x₂) :
  m < 1 := by sorry

end NUMINAMATH_CALUDE_double_points_imply_m_less_than_one_l357_35747


namespace NUMINAMATH_CALUDE_quadratic_roots_sequence_l357_35718

theorem quadratic_roots_sequence (p q a b : ℝ) : 
  p > 0 → 
  q > 0 → 
  a ≠ b → 
  a^2 - p*a + q = 0 → 
  b^2 - p*b + q = 0 → 
  ((a + b = 2*(-2) ∨ a + (-2) = 2*b ∨ b + (-2) = 2*a) ∧ 
   (a * b = (-2)^2 ∨ a * (-2) = b^2 ∨ b * (-2) = a^2)) → 
  p + q = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sequence_l357_35718


namespace NUMINAMATH_CALUDE_cube_volume_problem_l357_35737

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →
  a^3 - ((a-1)*(a-1)*(a+1)) = 7 →
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l357_35737


namespace NUMINAMATH_CALUDE_selling_price_formula_l357_35787

-- Define the relationship between quantity and selling price
def selling_price (x : ℕ+) : ℚ :=
  match x with
  | 1 => 8 + 0.3
  | 2 => 16 + 0.6
  | 3 => 24 + 0.9
  | 4 => 32 + 1.2
  | _ => 8.3 * x.val

-- Theorem statement
theorem selling_price_formula (x : ℕ+) :
  selling_price x = 8.3 * x.val := by sorry

end NUMINAMATH_CALUDE_selling_price_formula_l357_35787


namespace NUMINAMATH_CALUDE_h_inverse_at_one_l357_35717

def h (x : ℝ) : ℝ := 5 * x - 6

theorem h_inverse_at_one :
  ∃ b : ℝ, h b = 1 ∧ b = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_h_inverse_at_one_l357_35717


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l357_35731

/-- An isosceles right triangle with perimeter 10 + 10√2 has a hypotenuse of length 10 -/
theorem isosceles_right_triangle_hypotenuse (a c : ℝ) : 
  a > 0 → 
  c > 0 → 
  a^2 + a^2 = c^2 → 
  2*a + c = 10 + 10*Real.sqrt 2 → 
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l357_35731


namespace NUMINAMATH_CALUDE_burj_khalifa_sears_difference_l357_35748

/-- The height difference between two buildings -/
def height_difference (h1 h2 : ℕ) : ℕ := h1 - h2

/-- Burj Khalifa's height in meters -/
def burj_khalifa_height : ℕ := 830

/-- Sears Tower's height in meters -/
def sears_tower_height : ℕ := 527

/-- Theorem stating the height difference between Burj Khalifa and Sears Tower -/
theorem burj_khalifa_sears_difference :
  height_difference burj_khalifa_height sears_tower_height = 303 := by
  sorry

end NUMINAMATH_CALUDE_burj_khalifa_sears_difference_l357_35748


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_with_max_base_area_l357_35706

/-- The volume of a rectangular prism with maximum base area -/
theorem rectangular_prism_volume_with_max_base_area
  (height : ℝ)
  (base_circumference : ℝ)
  (h_height : height = 5)
  (h_base_circumference : base_circumference = 16)
  (h_base_max_area : ∀ (w l : ℝ), w + l = base_circumference / 2 → w * l ≤ (base_circumference / 4)^2) :
  height * (base_circumference / 4)^2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_with_max_base_area_l357_35706


namespace NUMINAMATH_CALUDE_buns_per_student_fourth_class_l357_35735

/-- Calculates the number of buns per student in the fourth class --/
def bunsPerStudentInFourthClass (
  numClasses : ℕ)
  (studentsPerClass : ℕ)
  (bunsPerPackage : ℕ)
  (packagesClass1 : ℕ)
  (packagesClass2 : ℕ)
  (packagesClass3 : ℕ)
  (packagesClass4 : ℕ)
  (staleBuns : ℕ)
  (uneatenBuns : ℕ) : ℕ :=
  let totalBunsClass4 := packagesClass4 * bunsPerPackage
  let totalUneatenBuns := staleBuns + uneatenBuns
  let uneatenBunsPerClass := totalUneatenBuns / numClasses
  let availableBunsClass4 := totalBunsClass4 - uneatenBunsPerClass
  availableBunsClass4 / studentsPerClass

/-- Theorem: Given the conditions, the number of buns per student in the fourth class is 9 --/
theorem buns_per_student_fourth_class :
  bunsPerStudentInFourthClass 4 30 8 20 25 30 35 16 20 = 9 := by
  sorry

end NUMINAMATH_CALUDE_buns_per_student_fourth_class_l357_35735


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l357_35722

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℤ := 2*n - 9
def S (n : ℕ) : ℤ := n^2 - 8*n

-- State the theorem
theorem arithmetic_sequence_properties :
  (a 1 = -7) ∧ 
  (S 3 = -15) ∧ 
  (∀ n : ℕ, S n = n^2 - 8*n) ∧
  (∃ min : ℤ, min = -16 ∧ ∀ n : ℕ, S n ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l357_35722


namespace NUMINAMATH_CALUDE_figure2_total_length_l357_35743

/-- The total length of segments in Figure 2 -/
def total_length (left right top bottom : ℝ) : ℝ :=
  left + right + top + bottom

/-- The theorem stating that the total length of segments in Figure 2 is 19 units -/
theorem figure2_total_length :
  ∃ (left right top bottom : ℝ),
    left = 8 ∧
    right = 6 ∧
    top = 1 ∧
    bottom = 2 + 1 + 1 ∧
    total_length left right top bottom = 19 := by
  sorry

end NUMINAMATH_CALUDE_figure2_total_length_l357_35743


namespace NUMINAMATH_CALUDE_min_parcels_covers_cost_l357_35765

/-- The minimum number of parcels Lucy must deliver to cover the cost of her motorbike -/
def min_parcels : ℕ := 750

/-- The cost of Lucy's motorbike -/
def motorbike_cost : ℕ := 6000

/-- Lucy's earnings per parcel -/
def earnings_per_parcel : ℕ := 12

/-- Lucy's fuel cost per delivery -/
def fuel_cost_per_delivery : ℕ := 4

/-- Theorem stating that min_parcels is the minimum number of parcels 
    Lucy must deliver to cover the cost of her motorbike -/
theorem min_parcels_covers_cost :
  (min_parcels * (earnings_per_parcel - fuel_cost_per_delivery) ≥ motorbike_cost) ∧
  ∀ n : ℕ, n < min_parcels → n * (earnings_per_parcel - fuel_cost_per_delivery) < motorbike_cost :=
sorry

end NUMINAMATH_CALUDE_min_parcels_covers_cost_l357_35765


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l357_35700

/-- Given a function f(x) = x^3 + ax^2 + 3x - 9 with an extreme value at x = -3, prove that a = 5 -/
theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + 3*x - 9
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ -3 → |x + 3| < ε → f x ≤ f (-3) ∨ f x ≥ f (-3)) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l357_35700


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l357_35729

theorem triangle_perimeter_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = π ∧  -- sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  a = 1 ∧  -- given condition
  a * Real.cos C + 0.5 * c = b →  -- given condition
  let l := a + b + c  -- perimeter definition
  2 < l ∧ l ≤ 3 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l357_35729


namespace NUMINAMATH_CALUDE_wandas_walk_l357_35738

/-- Proves that if Wanda walks 2 miles per day and 40 miles in 4 weeks, then she walks 5 days per week. -/
theorem wandas_walk (miles_per_day : ℝ) (total_miles : ℝ) (weeks : ℕ) (days_per_week : ℝ) : 
  miles_per_day = 2 → total_miles = 40 → weeks = 4 → 
  miles_per_day * days_per_week * weeks = total_miles → 
  days_per_week = 5 := by
sorry

end NUMINAMATH_CALUDE_wandas_walk_l357_35738


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l357_35709

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l357_35709


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l357_35713

/-- Represents the number of book types in each category -/
structure BookCategories where
  chinese : Nat
  mathematics : Nat
  liberal_arts : Nat
  english : Nat

/-- Calculates the total number of book types -/
def total_types (bc : BookCategories) : Nat :=
  bc.chinese + bc.mathematics + bc.liberal_arts + bc.english

/-- Calculates the number of books to be sampled from a category -/
def sample_size (category_size : Nat) (total : Nat) (sample : Nat) : Nat :=
  (category_size * sample) / total

theorem stratified_sample_sum (bc : BookCategories) (sample : Nat) :
  let total := total_types bc
  let math_sample := sample_size bc.mathematics total sample
  let liberal_arts_sample := sample_size bc.liberal_arts total sample
  bc.chinese = 20 →
  bc.mathematics = 10 →
  bc.liberal_arts = 40 →
  bc.english = 30 →
  sample = 20 →
  math_sample + liberal_arts_sample = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_l357_35713


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l357_35756

theorem right_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 17 → a = 15 →
  a^2 + b^2 = c^2 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l357_35756


namespace NUMINAMATH_CALUDE_marie_sells_40_loaves_l357_35733

/-- The number of loaves of bread Marie sells each day. -/
def L : ℕ := sorry

/-- The cost of the cash register in dollars. -/
def cash_register_cost : ℕ := 1040

/-- The price of each loaf of bread in dollars. -/
def bread_price : ℕ := 2

/-- The number of cakes sold daily. -/
def cakes_sold : ℕ := 6

/-- The price of each cake in dollars. -/
def cake_price : ℕ := 12

/-- The daily rent in dollars. -/
def daily_rent : ℕ := 20

/-- The daily electricity cost in dollars. -/
def daily_electricity : ℕ := 2

/-- The number of days needed to pay for the cash register. -/
def days_to_pay : ℕ := 8

/-- Theorem stating that Marie sells 40 loaves of bread each day. -/
theorem marie_sells_40_loaves : L = 40 := by
  sorry

end NUMINAMATH_CALUDE_marie_sells_40_loaves_l357_35733


namespace NUMINAMATH_CALUDE_prob_committee_with_both_genders_l357_35775

-- Define the total number of members
def total_members : ℕ := 40

-- Define the number of boys
def num_boys : ℕ := 18

-- Define the number of girls
def num_girls : ℕ := 22

-- Define the committee size
def committee_size : ℕ := 6

-- Define the probability function
noncomputable def prob_at_least_one_boy_one_girl : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) /
      (Nat.choose total_members committee_size : ℚ)

-- Theorem statement
theorem prob_committee_with_both_genders :
  prob_at_least_one_boy_one_girl = 2913683 / 3838380 :=
sorry

end NUMINAMATH_CALUDE_prob_committee_with_both_genders_l357_35775


namespace NUMINAMATH_CALUDE_minjun_height_l357_35715

/-- Calculates the current height given initial height and growth over two years -/
def current_height (initial : ℝ) (growth_last_year : ℝ) (growth_this_year : ℝ) : ℝ :=
  initial + growth_last_year + growth_this_year

/-- Theorem stating that Minjun's current height is 1.4 meters -/
theorem minjun_height :
  let initial_height : ℝ := 1.1
  let growth_last_year : ℝ := 0.2
  let growth_this_year : ℝ := 1/10
  current_height initial_height growth_last_year growth_this_year = 1.4 := by
  sorry

#eval current_height 1.1 0.2 0.1

end NUMINAMATH_CALUDE_minjun_height_l357_35715


namespace NUMINAMATH_CALUDE_ln_equation_solution_l357_35736

theorem ln_equation_solution :
  ∃ y : ℝ, (Real.log y - 3 * Real.log 2 = -1) ∧ (abs (y - 2.94) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ln_equation_solution_l357_35736


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l357_35776

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l357_35776


namespace NUMINAMATH_CALUDE_min_red_points_for_square_l357_35707

/-- A type representing a point on a circle -/
structure CirclePoint where
  angle : ℝ
  isOnCircle : 0 ≤ angle ∧ angle < 2 * π

/-- A function that determines if a point is colored red -/
def isRed : CirclePoint → Prop := sorry

/-- A predicate that checks if four points form a square on the circle -/
def formSquare (p1 p2 p3 p4 : CirclePoint) : Prop := sorry

/-- The theorem stating the minimum number of red points needed -/
theorem min_red_points_for_square (points : Fin 100 → CirclePoint)
  (equally_spaced : ∀ i : Fin 100, points i = ⟨2 * π * i / 100, sorry⟩) :
  (∃ red_points : Finset CirclePoint,
    red_points.card = 76 ∧
    (∀ p ∈ red_points, isRed p) ∧
    (∀ red_points' : Finset CirclePoint,
      red_points'.card < 76 →
      (∀ p ∈ red_points', isRed p) →
      ¬∃ p1 p2 p3 p4, formSquare p1 p2 p3 p4 ∧ isRed p1 ∧ isRed p2 ∧ isRed p3 ∧ isRed p4)) :=
sorry

end NUMINAMATH_CALUDE_min_red_points_for_square_l357_35707


namespace NUMINAMATH_CALUDE_combined_flock_size_l357_35708

def initial_flock : ℕ := 100
def net_increase_per_year : ℕ := 10
def years : ℕ := 5
def other_flock : ℕ := 150

theorem combined_flock_size :
  initial_flock + net_increase_per_year * years + other_flock = 300 := by
  sorry

end NUMINAMATH_CALUDE_combined_flock_size_l357_35708


namespace NUMINAMATH_CALUDE_tom_to_ben_ratio_l357_35766

def phillip_apples : ℕ := 40
def tom_apples : ℕ := 18
def ben_extra_apples : ℕ := 8

def ben_apples : ℕ := phillip_apples + ben_extra_apples

theorem tom_to_ben_ratio : 
  (tom_apples : ℚ) / (ben_apples : ℚ) = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_tom_to_ben_ratio_l357_35766


namespace NUMINAMATH_CALUDE_female_students_count_l357_35782

theorem female_students_count (female : ℕ) (male : ℕ) : 
  male = 3 * female →
  female + male = 52 →
  female = 13 := by
sorry

end NUMINAMATH_CALUDE_female_students_count_l357_35782


namespace NUMINAMATH_CALUDE_complex_root_equation_l357_35780

theorem complex_root_equation (z : ℂ) : 
  (∃ a b : ℝ, z = a + b * I) → 
  z^2 = -3 - 4 * I → 
  z = -1 + 2 * I := by sorry

end NUMINAMATH_CALUDE_complex_root_equation_l357_35780


namespace NUMINAMATH_CALUDE_faucet_flow_rate_l357_35759

/-- The flow rate of a faucet given the number of barrels, capacity per barrel, and time to fill --/
def flowRate (numBarrels : ℕ) (capacityPerBarrel : ℚ) (timeToFill : ℚ) : ℚ :=
  (numBarrels : ℚ) * capacityPerBarrel / timeToFill

/-- Theorem stating that the flow rate for the given conditions is 3.5 gallons per minute --/
theorem faucet_flow_rate :
  flowRate 4 7 8 = (7/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_faucet_flow_rate_l357_35759


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l357_35751

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- A line with a given distance from a point -/
structure Line where
  distanceFromPoint : ℝ

/-- Defines the relationship between a line and a circle -/
inductive Relationship
  | Intersecting
  | Tangent
  | Disjoint

theorem line_tangent_to_circle (c : Circle) (l : Line) :
  c.diameter = 13 →
  l.distanceFromPoint = 13 / 2 →
  Relationship.Tangent = (
    if l.distanceFromPoint = c.diameter / 2
    then Relationship.Tangent
    else if l.distanceFromPoint < c.diameter / 2
    then Relationship.Intersecting
    else Relationship.Disjoint
  ) := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l357_35751


namespace NUMINAMATH_CALUDE_iterative_insertion_square_l357_35785

theorem iterative_insertion_square (n : ℕ) : ∃ m : ℕ, 
  4 * (10^n - 1) / 9 * 10^(n-1) + 8 * (10^(n-1) - 1) / 9 + 9 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_iterative_insertion_square_l357_35785


namespace NUMINAMATH_CALUDE_circle_equation_l357_35750

/-- A circle C with center (a, 0) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ

/-- The line l: y = 2x + 1 -/
def line (x : ℝ) : ℝ := 2 * x + 1

/-- The point P(0, 1) -/
def P : ℝ × ℝ := (0, 1)

/-- The circle C is tangent to the line l at point P -/
def is_tangent (C : Circle) : Prop :=
  C.r^2 = (C.a - P.1)^2 + (0 - P.2)^2 ∧
  (0 - P.2) / (C.a - P.1) = -1 / 2

theorem circle_equation (C : Circle) 
  (h1 : is_tangent C) : 
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 5 ↔ (x - C.a)^2 + y^2 = C.r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l357_35750


namespace NUMINAMATH_CALUDE_darwin_money_left_l357_35727

theorem darwin_money_left (initial_amount : ℚ) (gas_fraction : ℚ) (food_fraction : ℚ) 
  (h1 : initial_amount = 600)
  (h2 : gas_fraction = 1/3)
  (h3 : food_fraction = 1/4) :
  initial_amount - (gas_fraction * initial_amount) - (food_fraction * (initial_amount - (gas_fraction * initial_amount))) = 300 := by
  sorry

end NUMINAMATH_CALUDE_darwin_money_left_l357_35727


namespace NUMINAMATH_CALUDE_heptagon_sum_l357_35726

/-- Represents a heptagon with numbers distributed on its sides -/
structure NumberedHeptagon where
  /-- Total number of circles in the heptagon -/
  total_circles : Nat
  /-- Number of circles on each side of the heptagon -/
  circles_per_side : Nat
  /-- Total number of sides in the heptagon -/
  sides : Nat
  /-- The sum of all numbers distributed in the heptagon -/
  total_sum : Nat
  /-- The sum of numbers 1 to 7 -/
  sum_1_to_7 : Nat
  /-- Condition: Total circles is the product of circles per side and number of sides -/
  h_total : total_circles = circles_per_side * sides
  /-- Condition: The heptagon has 7 sides -/
  h_heptagon : sides = 7
  /-- Condition: Each side has 3 circles -/
  h_three_per_side : circles_per_side = 3
  /-- Condition: The total sum is the sum of numbers 1 to 14 plus the sum of 1 to 7 -/
  h_sum : total_sum = (14 * 15) / 2 + sum_1_to_7
  /-- Condition: The sum of numbers 1 to 7 -/
  h_sum_1_to_7 : sum_1_to_7 = (7 * 8) / 2

/-- Theorem: The sum of numbers in each line of three circles is 19 -/
theorem heptagon_sum (h : NumberedHeptagon) : h.total_sum / h.sides = 19 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_sum_l357_35726


namespace NUMINAMATH_CALUDE_diameter_circle_equation_l357_35761

/-- A circle passing through two points, where the line segment between the points is a diameter -/
structure DiameterCircle where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The standard equation of a circle -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the given circle has the specified equation -/
theorem diameter_circle_equation (C : DiameterCircle) 
  (h₁ : C.A = (1, 2)) 
  (h₂ : C.B = (3, 1)) : 
  ∀ x y, circle_equation 2 (3/2) (5/4) x y :=
sorry

end NUMINAMATH_CALUDE_diameter_circle_equation_l357_35761


namespace NUMINAMATH_CALUDE_rationality_of_expressions_l357_35716

theorem rationality_of_expressions :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt (Real.exp 2) = p / q) ∧
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ (0.64 : ℝ) ^ (1/3) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (0.0256 : ℝ) ^ (1/4) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-8 : ℝ) ^ (1/3) * Real.sqrt ((0.25 : ℝ)⁻¹) = p / q) :=
by sorry

end NUMINAMATH_CALUDE_rationality_of_expressions_l357_35716


namespace NUMINAMATH_CALUDE_quilt_cost_is_450_l357_35712

/-- Calculates the total cost of patches for a quilt with given dimensions and pricing rules. -/
def quilt_patch_cost (quilt_length : ℕ) (quilt_width : ℕ) (patch_area : ℕ) 
                     (first_patch_cost : ℕ) (first_patch_count : ℕ) : ℕ :=
  let total_area := quilt_length * quilt_width
  let total_patches := total_area / patch_area
  let remaining_patches := total_patches - first_patch_count
  let first_batch_cost := first_patch_count * first_patch_cost
  let remaining_cost := remaining_patches * (first_patch_cost / 2)
  first_batch_cost + remaining_cost

/-- Proves that the total cost of patches for the specified quilt is $450. -/
theorem quilt_cost_is_450 : quilt_patch_cost 16 20 4 10 10 = 450 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_is_450_l357_35712


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l357_35720

/-- Represents a monthly bill for an online service provider -/
structure MonthlyBill where
  fixed_fee : ℝ
  hourly_rate : ℝ
  hours_used : ℝ

/-- Calculates the total bill amount -/
def MonthlyBill.total (bill : MonthlyBill) : ℝ :=
  bill.fixed_fee + bill.hourly_rate * bill.hours_used

theorem fixed_fee_calculation (feb_bill mar_bill : MonthlyBill) 
  (h1 : feb_bill.total = 20.72)
  (h2 : mar_bill.total = 35.28)
  (h3 : feb_bill.fixed_fee = mar_bill.fixed_fee)
  (h4 : feb_bill.hourly_rate = mar_bill.hourly_rate)
  (h5 : mar_bill.hours_used = 3 * feb_bill.hours_used) :
  feb_bill.fixed_fee = 13.44 := by
  sorry

#check fixed_fee_calculation

end NUMINAMATH_CALUDE_fixed_fee_calculation_l357_35720


namespace NUMINAMATH_CALUDE_house_cleaning_time_l357_35770

def total_time (dawn_dish_time andy_laundry_time andy_vacuum_time dawn_window_time : ℝ) : ℝ :=
  dawn_dish_time + andy_laundry_time + andy_vacuum_time + dawn_window_time

theorem house_cleaning_time : ∃ (dawn_dish_time andy_laundry_time andy_vacuum_time dawn_window_time : ℝ),
  dawn_dish_time = 20 ∧
  andy_laundry_time = 2 * dawn_dish_time + 6 ∧
  andy_vacuum_time = Real.sqrt (andy_laundry_time - dawn_dish_time) ∧
  dawn_window_time = (1 / 4) * (andy_laundry_time + dawn_dish_time) ∧
  total_time dawn_dish_time andy_laundry_time andy_vacuum_time dawn_window_time = 87.6 := by
  sorry

end NUMINAMATH_CALUDE_house_cleaning_time_l357_35770


namespace NUMINAMATH_CALUDE_set_operations_l357_35763

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}

theorem set_operations :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l357_35763


namespace NUMINAMATH_CALUDE_no_valid_c_l357_35755

theorem no_valid_c : ¬ ∃ (c : ℕ+), 
  (∃ (x y : ℚ), 3 * x^2 + 7 * x + c.val = 0 ∧ 3 * y^2 + 7 * y + c.val = 0 ∧ x ≠ y) ∧ 
  (∃ (x y : ℚ), 3 * x^2 + 7 * y + c.val = 0 ∧ 3 * y^2 + 7 * y + c.val = 0 ∧ x + y > 4) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_c_l357_35755


namespace NUMINAMATH_CALUDE_missing_number_solution_l357_35799

theorem missing_number_solution : ∃ x : ℤ, 10010 - 12 * x * 2 = 9938 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_solution_l357_35799


namespace NUMINAMATH_CALUDE_max_value_of_f_l357_35746

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (8 - x))

theorem max_value_of_f :
  ∃ (M : ℝ) (x₀ : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 8 → f x ≤ M) ∧
    (0 ≤ x₀ ∧ x₀ ≤ 8) ∧
    (f x₀ = M) ∧
    (x₀ = 200 / 27) ∧
    (M = 12 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l357_35746


namespace NUMINAMATH_CALUDE_business_profit_calculation_l357_35739

theorem business_profit_calculation (suresh_investment ramesh_investment ramesh_profit_share : ℕ) :
  suresh_investment = 24000 →
  ramesh_investment = 40000 →
  ramesh_profit_share = 11875 →
  ∃ (total_profit : ℕ), total_profit = 19000 :=
by sorry

end NUMINAMATH_CALUDE_business_profit_calculation_l357_35739


namespace NUMINAMATH_CALUDE_find_m_l357_35784

theorem find_m : ∃ m : ℝ, ∀ x : ℝ, (x - 4) * (x + 3) = x^2 + m*x - 12 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l357_35784


namespace NUMINAMATH_CALUDE_cubic_equation_root_implies_m_range_l357_35795

theorem cubic_equation_root_implies_m_range :
  ∀ m : ℝ, 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^3 - 3*x + m = 0) →
  m ∈ Set.Icc (-2) 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_implies_m_range_l357_35795


namespace NUMINAMATH_CALUDE_borrowed_with_interest_l357_35754

/-- The amount to be returned after borrowing with interest -/
def amount_to_return (borrowed : ℝ) (interest_rate : ℝ) : ℝ :=
  borrowed * (1 + interest_rate)

/-- Theorem: Borrowing $100 with 10% interest results in returning $110 -/
theorem borrowed_with_interest :
  amount_to_return 100 0.1 = 110 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_with_interest_l357_35754


namespace NUMINAMATH_CALUDE_congruence_solution_l357_35710

theorem congruence_solution (x : ℤ) : 
  (10 * x + 3) % 18 = 7 % 18 → x % 9 = 4 % 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l357_35710


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l357_35757

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (sum_interior_angles octagon_sides) / octagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l357_35757


namespace NUMINAMATH_CALUDE_binary_11101_is_29_l357_35723

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldr (fun (i, b) acc => acc + if b then 2^i else 0) 0

/-- The binary representation of the number in question -/
def binary_number : List Bool := [true, false, true, true, true]

/-- Theorem stating that the decimal representation of 11101₂ is 29 -/
theorem binary_11101_is_29 : binary_to_decimal binary_number = 29 := by
  sorry

end NUMINAMATH_CALUDE_binary_11101_is_29_l357_35723


namespace NUMINAMATH_CALUDE_iris_shopping_expense_l357_35704

def jacket_price : ℕ := 10
def shorts_price : ℕ := 6
def pants_price : ℕ := 12

def jacket_quantity : ℕ := 3
def shorts_quantity : ℕ := 2
def pants_quantity : ℕ := 4

def total_spent : ℕ := jacket_price * jacket_quantity + shorts_price * shorts_quantity + pants_price * pants_quantity

theorem iris_shopping_expense : total_spent = 90 := by
  sorry

end NUMINAMATH_CALUDE_iris_shopping_expense_l357_35704


namespace NUMINAMATH_CALUDE_m_necessary_not_sufficient_for_n_l357_35796

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem stating the relationship between M and N
theorem m_necessary_not_sufficient_for_n :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_m_necessary_not_sufficient_for_n_l357_35796


namespace NUMINAMATH_CALUDE_pillars_count_l357_35764

/-- The length of the circular track in meters -/
def track_length : ℕ := 1200

/-- The interval between pillars in meters -/
def pillar_interval : ℕ := 30

/-- The number of pillars along the circular track -/
def num_pillars : ℕ := track_length / pillar_interval

theorem pillars_count : num_pillars = 40 := by
  sorry

end NUMINAMATH_CALUDE_pillars_count_l357_35764


namespace NUMINAMATH_CALUDE_display_window_configurations_l357_35724

theorem display_window_configurations :
  let fiction_books : ℕ := 3
  let nonfiction_books : ℕ := 3
  let fiction_arrangements := Nat.factorial fiction_books
  let nonfiction_arrangements := Nat.factorial nonfiction_books
  fiction_arrangements * nonfiction_arrangements = 36 :=
by sorry

end NUMINAMATH_CALUDE_display_window_configurations_l357_35724


namespace NUMINAMATH_CALUDE_gcd_98_63_l357_35741

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l357_35741


namespace NUMINAMATH_CALUDE_matt_problem_time_l357_35786

/-- The time it takes Matt to do a problem without a calculator -/
def time_without_calculator : ℝ := sorry

/-- The time it takes Matt to do a problem with a calculator -/
def time_with_calculator : ℝ := 2

/-- The number of problems in Matt's assignment -/
def number_of_problems : ℕ := 20

/-- The total time saved by using a calculator -/
def time_saved : ℝ := 60

theorem matt_problem_time :
  time_without_calculator = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_matt_problem_time_l357_35786


namespace NUMINAMATH_CALUDE_mrs_hilt_books_l357_35778

/-- The number of books Mrs. Hilt bought -/
def num_books : ℕ := sorry

/-- The cost per book when buying -/
def cost_per_book : ℕ := 11

/-- The price per book when selling -/
def price_per_book : ℕ := 25

/-- The difference between total sold amount and total paid amount -/
def profit : ℕ := 210

theorem mrs_hilt_books :
  num_books * price_per_book - num_books * cost_per_book = profit ∧ num_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_books_l357_35778


namespace NUMINAMATH_CALUDE_joan_remaining_flour_l357_35767

/-- Given a cake recipe that requires a certain amount of flour and the amount already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Theorem: Joan needs to add 4 more cups of flour. -/
theorem joan_remaining_flour :
  remaining_flour 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_flour_l357_35767


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l357_35719

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (sum_eq : x + y = 3 * x * y) (diff_eq : x - y = 2) : 
  1 / x + 1 / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l357_35719


namespace NUMINAMATH_CALUDE_min_filtrations_correct_l357_35792

/-- The initial concentration of pollutants in mg/cm³ -/
def initial_concentration : ℝ := 1.2

/-- The reduction factor for each filtration -/
def reduction_factor : ℝ := 0.8

/-- The target concentration of pollutants in mg/cm³ -/
def target_concentration : ℝ := 0.2

/-- The minimum number of filtrations needed to reach the target concentration -/
def min_filtrations : ℕ := 8

theorem min_filtrations_correct :
  (∀ n : ℕ, n < min_filtrations → initial_concentration * reduction_factor ^ n > target_concentration) ∧
  initial_concentration * reduction_factor ^ min_filtrations ≤ target_concentration :=
by sorry

end NUMINAMATH_CALUDE_min_filtrations_correct_l357_35792


namespace NUMINAMATH_CALUDE_counterexample_dot_product_equality_l357_35714

theorem counterexample_dot_product_equality :
  ∃ (a b c : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ c ≠ (0, 0) ∧
  (a.1 * b.1 + a.2 * b.2 = a.1 * c.1 + a.2 * c.2) ∧ b ≠ c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_dot_product_equality_l357_35714


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l357_35758

/-- Given the total selling price, profit per meter, and cost price per meter of cloth,
    calculate the number of meters sold. -/
theorem cloth_sale_calculation (total_selling_price profit_per_meter cost_price_per_meter : ℚ) :
  total_selling_price = 10000 ∧ 
  profit_per_meter = 7 ∧ 
  cost_price_per_meter = 118 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℚ) = 80 := by
  sorry

#eval (10000 : ℚ) / (118 + 7) -- This should evaluate to 80

end NUMINAMATH_CALUDE_cloth_sale_calculation_l357_35758


namespace NUMINAMATH_CALUDE_big_stack_pancakes_l357_35744

/-- The number of pancakes in a big stack at Hank's cafe. -/
def big_stack : ℕ := sorry

/-- The number of pancakes in a short stack at Hank's cafe. -/
def short_stack : ℕ := 3

/-- The number of customers who ordered short stack. -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack. -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes Hank needs to make. -/
def total_pancakes : ℕ := 57

/-- Theorem stating that the number of pancakes in a big stack is 5. -/
theorem big_stack_pancakes : 
  short_stack * short_stack_orders + big_stack * big_stack_orders = total_pancakes → 
  big_stack = 5 := by sorry

end NUMINAMATH_CALUDE_big_stack_pancakes_l357_35744


namespace NUMINAMATH_CALUDE_existence_of_more_good_numbers_l357_35772

/-- A function that determines if a natural number is "good" or "bad" --/
def isGoodNumber (x : ℕ) : Bool := sorry

/-- The count of n-digit numbers that are "good" --/
def countGoodNumbers (n : ℕ) : ℕ := sorry

/-- The count of n-digit numbers that are "bad" --/
def countBadNumbers (n : ℕ) : ℕ := sorry

/-- The total count of n-digit numbers --/
def totalNumbers (n : ℕ) : ℕ := 9 * (10 ^ (n - 1))

theorem existence_of_more_good_numbers :
  ∃ n : ℕ, n ≥ 4 ∧ countGoodNumbers n > countBadNumbers n :=
sorry

end NUMINAMATH_CALUDE_existence_of_more_good_numbers_l357_35772


namespace NUMINAMATH_CALUDE_intersection_product_l357_35783

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 4

-- Define the line m in polar coordinates
def line_m (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 3) = 2

-- Define the ray l in polar coordinates
def ray_l (ρ θ : ℝ) : Prop := θ = 5 * Real.pi / 6 ∧ ρ ≥ 0

-- Theorem statement
theorem intersection_product :
  ∃ (ρ_A ρ_B : ℝ),
    (∃ (x_A y_A : ℝ), circle_C x_A y_A ∧ x_A = ρ_A * Real.cos (5 * Real.pi / 6) ∧ y_A = ρ_A * Real.sin (5 * Real.pi / 6)) ∧
    (line_m ρ_B (5 * Real.pi / 6)) ∧
    ρ_A * ρ_B = -3 + Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_intersection_product_l357_35783


namespace NUMINAMATH_CALUDE_locus_of_centers_l357_35798

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16

-- Define the property of being externally tangent to C1 and internally tangent to C2
def externally_internally_tangent (a b r : ℝ) : Prop :=
  (∃ x y : ℝ, C1 x y ∧ (a - x)^2 + (b - y)^2 = (r + 1)^2) ∧
  (∃ x y : ℝ, C2 x y ∧ (a - x)^2 + (b - y)^2 = (4 - r)^2)

-- State the theorem
theorem locus_of_centers :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_internally_tangent a b r) ↔
  84 * a^2 + 100 * b^2 - 168 * a - 441 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l357_35798


namespace NUMINAMATH_CALUDE_complement_of_union_l357_35791

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l357_35791


namespace NUMINAMATH_CALUDE_power_of_product_l357_35730

/-- For all real numbers m and n, (2m²n)³ = 8m⁶n³ -/
theorem power_of_product (m n : ℝ) : (2 * m^2 * n)^3 = 8 * m^6 * n^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l357_35730


namespace NUMINAMATH_CALUDE_one_multiple_choice_one_true_false_prob_at_least_one_multiple_choice_prob_l357_35773

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 20

-- Theorem for the first probability
theorem one_multiple_choice_one_true_false_prob :
  (multiple_choice_questions * true_false_questions * 2) / total_outcomes = 3 / 5 := by
  sorry

-- Theorem for the second probability
theorem at_least_one_multiple_choice_prob :
  1 - (true_false_questions * (true_false_questions - 1) / 2) / total_outcomes = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_multiple_choice_one_true_false_prob_at_least_one_multiple_choice_prob_l357_35773


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l357_35760

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

theorem intersection_of_M_and_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l357_35760


namespace NUMINAMATH_CALUDE_freddy_calls_cost_l357_35734

/-- Calculates the total cost of phone calls in dollars -/
def total_cost_dollars (local_rate : ℚ) (intl_rate : ℚ) (local_duration : ℕ) (intl_duration : ℕ) : ℚ :=
  ((local_rate * local_duration + intl_rate * intl_duration) / 100 : ℚ)

theorem freddy_calls_cost :
  let local_rate : ℚ := 5
  let intl_rate : ℚ := 25
  let local_duration : ℕ := 45
  let intl_duration : ℕ := 31
  total_cost_dollars local_rate intl_rate local_duration intl_duration = 10 := by
sorry

#eval total_cost_dollars 5 25 45 31

end NUMINAMATH_CALUDE_freddy_calls_cost_l357_35734


namespace NUMINAMATH_CALUDE_distance_is_8_sqrt_2_l357_35725

/-- Two externally tangent circles with a common external tangent -/
structure TangentCircles where
  /-- Radius of the larger circle -/
  r₁ : ℝ
  /-- Radius of the smaller circle -/
  r₂ : ℝ
  /-- The circles are externally tangent -/
  tangent : r₁ > r₂
  /-- The radii are 8 and 2 units respectively -/
  radius_values : r₁ = 8 ∧ r₂ = 2

/-- The distance from the center of the larger circle to the point where 
    the common external tangent touches the smaller circle -/
def distance_to_tangent_point (c : TangentCircles) : ℝ :=
  sorry

/-- Theorem stating that the distance is 8√2 -/
theorem distance_is_8_sqrt_2 (c : TangentCircles) : 
  distance_to_tangent_point c = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_8_sqrt_2_l357_35725


namespace NUMINAMATH_CALUDE_g_3_value_l357_35777

/-- A linear function satisfying certain conditions -/
def g (x : ℝ) : ℝ := sorry

/-- The inverse of g -/
def g_inv (x : ℝ) : ℝ := sorry

/-- g is a linear function -/
axiom g_linear : ∃ (c d : ℝ), ∀ x, g x = c * x + d

/-- g satisfies the equation g(x) = 5g^(-1)(x) + 3 -/
axiom g_equation : ∀ x, g x = 5 * g_inv x + 3

/-- g(2) = 5 -/
axiom g_2_eq_5 : g 2 = 5

/-- Main theorem: g(3) = 3√5 + (3√5)/(√5 + 5) -/
theorem g_3_value : g 3 = 3 * Real.sqrt 5 + (3 * Real.sqrt 5) / (Real.sqrt 5 + 5) := by sorry

end NUMINAMATH_CALUDE_g_3_value_l357_35777


namespace NUMINAMATH_CALUDE_root_transformation_l357_35781

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 9 = 0) → 
  ((3*r₁)^3 - 12*(3*r₁)^2 + 243 = 0) ∧ 
  ((3*r₂)^3 - 12*(3*r₂)^2 + 243 = 0) ∧ 
  ((3*r₃)^3 - 12*(3*r₃)^2 + 243 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l357_35781
