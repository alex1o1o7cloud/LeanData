import Mathlib

namespace NUMINAMATH_CALUDE_cos_function_identity_l3847_384737

theorem cos_function_identity (f : ℝ → ℝ) (x : ℝ) 
  (h : ∀ x, f (Real.sin x) = 2 - Real.cos (2 * x)) : 
  f (Real.cos x) = 2 + Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_function_identity_l3847_384737


namespace NUMINAMATH_CALUDE_find_divisor_l3847_384748

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 997) (h2 : quotient = 43) (h3 : remainder = 8) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 23 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3847_384748


namespace NUMINAMATH_CALUDE_equation_solution_l3847_384795

theorem equation_solution : ∃! x : ℝ, 0.05 * x + 0.12 * (30 + x) = 15.84 ∧ x = 72 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3847_384795


namespace NUMINAMATH_CALUDE_complex_magnitude_eval_l3847_384754

theorem complex_magnitude_eval (ω : ℂ) (h : ω = 7 + 3 * I) :
  Complex.abs (ω^2 + 8*ω + 85) = Real.sqrt 30277 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_eval_l3847_384754


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3847_384701

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3847_384701


namespace NUMINAMATH_CALUDE_total_fishes_is_32_l3847_384716

/-- The total number of fishes caught by Melanie and Tom -/
def total_fishes (melanie_trout : ℕ) (tom_salmon_multiplier : ℕ) : ℕ :=
  melanie_trout + tom_salmon_multiplier * melanie_trout

/-- Proof that the total number of fishes caught is 32 -/
theorem total_fishes_is_32 : total_fishes 8 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_fishes_is_32_l3847_384716


namespace NUMINAMATH_CALUDE_seedling_problem_l3847_384729

/-- Represents the unit price and quantity of seedlings --/
structure Seedling where
  price : ℚ
  quantity : ℚ

/-- Represents the total cost of a purchase --/
def totalCost (a b : Seedling) : ℚ :=
  a.price * a.quantity + b.price * b.quantity

/-- Represents the discounted price of a seedling --/
def discountedPrice (s : Seedling) (discount : ℚ) : ℚ :=
  s.price * (1 - discount)

theorem seedling_problem :
  ∃ (a b : Seedling),
    (totalCost ⟨a.price, 15⟩ ⟨b.price, 5⟩ = 190) ∧
    (totalCost ⟨a.price, 25⟩ ⟨b.price, 15⟩ = 370) ∧
    (a.price = 10) ∧
    (b.price = 8) ∧
    (∀ m : ℚ,
      m ≤ 100 ∧
      (discountedPrice a 0.1) * m + (discountedPrice b 0.1) * (100 - m) ≤ 828 →
      m ≤ 60) ∧
    (∃ m : ℚ,
      m = 60 ∧
      (discountedPrice a 0.1) * m + (discountedPrice b 0.1) * (100 - m) ≤ 828) :=
by
  sorry


end NUMINAMATH_CALUDE_seedling_problem_l3847_384729


namespace NUMINAMATH_CALUDE_button_sequence_l3847_384788

theorem button_sequence (a : Fin 6 → ℕ) (h1 : a 0 = 1)
    (h2 : a 1 = 3) (h4 : a 3 = 27) (h5 : a 4 = 81) (h6 : a 5 = 243)
    (h_ratio : ∀ i : Fin 5, a (i + 1) = 3 * a i) : a 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_button_sequence_l3847_384788


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_l3847_384735

theorem cos_alpha_minus_pi (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) : 
  Real.cos (α - π) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_l3847_384735


namespace NUMINAMATH_CALUDE_power_product_evaluation_l3847_384782

theorem power_product_evaluation (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l3847_384782


namespace NUMINAMATH_CALUDE_sector_max_area_l3847_384756

/-- Given a sector of a circle with radius R, central angle α, and fixed perimeter c,
    the maximum area of the sector is c²/16. -/
theorem sector_max_area (R α c : ℝ) (h_pos_R : R > 0) (h_pos_α : α > 0) (h_pos_c : c > 0)
  (h_perimeter : c = 2 * R + R * α) :
  ∃ (A : ℝ), A ≤ c^2 / 16 ∧ 
  (∀ (R' α' : ℝ), R' > 0 → α' > 0 → c = 2 * R' + R' * α' → 
    (1/2) * R' * R' * α' ≤ A) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3847_384756


namespace NUMINAMATH_CALUDE_f_3_equals_11_l3847_384769

-- Define the function f
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x + 2

-- State the theorem
theorem f_3_equals_11 (a b : ℝ) :
  f 1 a b = 5 →
  f 2 a b = 8 →
  f 3 a b = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_11_l3847_384769


namespace NUMINAMATH_CALUDE_rectangle_area_l3847_384789

/-- Given a rectangle with perimeter 280 meters and length-to-width ratio of 5:2, its area is 4000 square meters. -/
theorem rectangle_area (L W : ℝ) (h1 : 2*L + 2*W = 280) (h2 : L / W = 5 / 2) : L * W = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3847_384789


namespace NUMINAMATH_CALUDE_evaluate_expression_l3847_384752

/-- Given x = 4 and z = -2, prove that z(z - 4x) = 36 -/
theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3847_384752


namespace NUMINAMATH_CALUDE_only_2020_is_very_good_l3847_384741

/-- Represents a four-digit number YEAR --/
structure Year where
  Y : Fin 10
  E : Fin 10
  A : Fin 10
  R : Fin 10

/-- Checks if a Year is in the 21st century --/
def is_21st_century (year : Year) : Prop :=
  2001 ≤ year.Y * 1000 + year.E * 100 + year.A * 10 + year.R ∧ 
  year.Y * 1000 + year.E * 100 + year.A * 10 + year.R ≤ 2100

/-- The system of linear equations for a given Year --/
def system_has_multiple_solutions (year : Year) : Prop :=
  ∃ (x y z w : ℝ) (x' y' z' w' : ℝ),
    (x ≠ x' ∨ y ≠ y' ∨ z ≠ z' ∨ w ≠ w') ∧
    (year.Y * x + year.E * y + year.A * z + year.R * w = year.Y) ∧
    (year.R * x + year.Y * y + year.E * z + year.A * w = year.E) ∧
    (year.A * x + year.R * y + year.Y * z + year.E * w = year.A) ∧
    (year.E * x + year.A * y + year.R * z + year.Y * w = year.R) ∧
    (year.Y * x' + year.E * y' + year.A * z' + year.R * w' = year.Y) ∧
    (year.R * x' + year.Y * y' + year.E * z' + year.A * w' = year.E) ∧
    (year.A * x' + year.R * y' + year.Y * z' + year.E * w' = year.A) ∧
    (year.E * x' + year.A * y' + year.R * z' + year.Y * w' = year.R)

/-- The main theorem stating that 2020 is the only "very good" year in the 21st century --/
theorem only_2020_is_very_good :
  ∀ (year : Year),
    is_21st_century year ∧ system_has_multiple_solutions year ↔
    year.Y = 2 ∧ year.E = 0 ∧ year.A = 2 ∧ year.R = 0 :=
sorry

end NUMINAMATH_CALUDE_only_2020_is_very_good_l3847_384741


namespace NUMINAMATH_CALUDE_infinite_solutions_when_m_is_two_l3847_384759

theorem infinite_solutions_when_m_is_two :
  ∃ (m : ℝ), ∀ (x : ℝ), m^2 * x + m * (1 - x) - 2 * (1 + x) = 0 → 
  (m = 2 ∧ ∀ (y : ℝ), m^2 * y + m * (1 - y) - 2 * (1 + y) = 0) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_when_m_is_two_l3847_384759


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l3847_384766

def total_money : ℚ := 90
def sandwich_fraction : ℚ := 1/5
def book_fraction : ℚ := 1/2
def money_left : ℚ := 12

theorem museum_ticket_fraction :
  let spent := total_money - money_left
  let sandwich_cost := sandwich_fraction * total_money
  let book_cost := book_fraction * total_money
  let museum_cost := spent - (sandwich_cost + book_cost)
  museum_cost / total_money = 1/6 := by sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l3847_384766


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3847_384708

theorem sin_2alpha_value (α : Real) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3847_384708


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3847_384775

def coin_flip_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem unfair_coin_flip_probability :
  let n : ℕ := 8  -- Total number of flips
  let k : ℕ := 3  -- Number of tails
  let p : ℚ := 2/3  -- Probability of tails
  coin_flip_probability n k p = 448/6561 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l3847_384775


namespace NUMINAMATH_CALUDE_divisibility_of_repeated_eight_l3847_384746

theorem divisibility_of_repeated_eight : ∃ k : ℕ, 8 * (10^1974 - 1) / 9 = 13 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_repeated_eight_l3847_384746


namespace NUMINAMATH_CALUDE_watch_cost_price_proof_l3847_384726

/-- The cost price of a watch satisfying certain selling conditions -/
def watch_cost_price : ℝ := 875

/-- The selling price of the watch at a loss -/
def selling_price_loss : ℝ := watch_cost_price * (1 - 0.12)

/-- The selling price of the watch at a gain -/
def selling_price_gain : ℝ := watch_cost_price * (1 + 0.04)

/-- Theorem stating the cost price of the watch given the selling conditions -/
theorem watch_cost_price_proof :
  (selling_price_loss = watch_cost_price * (1 - 0.12)) ∧
  (selling_price_gain = watch_cost_price * (1 + 0.04)) ∧
  (selling_price_gain - selling_price_loss = 140) →
  watch_cost_price = 875 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_proof_l3847_384726


namespace NUMINAMATH_CALUDE_factoring_expression_l3847_384727

theorem factoring_expression (x : ℝ) : 2 * x * (x + 3) + 4 * (x + 3) = 2 * (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3847_384727


namespace NUMINAMATH_CALUDE_system_solution_l3847_384731

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop :=
  7 / (2 * x - 3) - 2 / (10 * z - 3 * y) + 3 / (3 * y - 8 * z) = 8

def equation2 (x y z : ℝ) : Prop :=
  2 / (2 * x - 3 * y) - 3 / (10 * z - 3 * y) + 1 / (3 * y - 8 * z) = 0

def equation3 (x y z : ℝ) : Prop :=
  5 / (2 * x - 3 * y) - 4 / (10 * z - 3 * y) + 7 / (3 * y - 8 * z) = 8

-- Define the solution
def solution : ℝ × ℝ × ℝ := (5, 3, 1)

-- Theorem statement
theorem system_solution :
  ∀ x y z : ℝ,
  2 * x ≠ 3 * y →
  10 * z ≠ 3 * y →
  8 * z ≠ 3 * y →
  equation1 x y z ∧ equation2 x y z ∧ equation3 x y z →
  (x, y, z) = solution :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3847_384731


namespace NUMINAMATH_CALUDE_robot_capacity_theorem_l3847_384796

/-- Represents the material handling capacity of robots A and B --/
structure RobotCapacity where
  A : ℝ
  B : ℝ

/-- The conditions given in the problem --/
def satisfiesConditions (c : RobotCapacity) : Prop :=
  c.A = c.B + 30 ∧ 1000 / c.A = 800 / c.B

/-- The theorem to prove --/
theorem robot_capacity_theorem :
  ∃ c : RobotCapacity, satisfiesConditions c ∧ c.A = 150 ∧ c.B = 120 := by
  sorry

end NUMINAMATH_CALUDE_robot_capacity_theorem_l3847_384796


namespace NUMINAMATH_CALUDE_curler_count_l3847_384784

theorem curler_count (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ) : 
  total = 16 →
  pink = total / 4 →
  blue = 2 * pink →
  green = total - (pink + blue) →
  green = 4 := by
  sorry

end NUMINAMATH_CALUDE_curler_count_l3847_384784


namespace NUMINAMATH_CALUDE_min_both_beethoven_vivaldi_l3847_384747

/-- The minimum number of people who like both Beethoven and Vivaldi in a group of 120 people,
    where 95 like Beethoven and 80 like Vivaldi. -/
theorem min_both_beethoven_vivaldi (total : ℕ) (beethoven : ℕ) (vivaldi : ℕ)
    (h_total : total = 120)
    (h_beethoven : beethoven = 95)
    (h_vivaldi : vivaldi = 80) :
    beethoven + vivaldi - total ≥ 55 := by
  sorry

end NUMINAMATH_CALUDE_min_both_beethoven_vivaldi_l3847_384747


namespace NUMINAMATH_CALUDE_photo_arrangements_l3847_384740

def number_of_students : ℕ := 7
def number_of_bound_students : ℕ := 2
def number_of_separated_students : ℕ := 2

def arrangements (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem photo_arrangements :
  let bound_ways := number_of_bound_students
  let remaining_elements := number_of_students - number_of_bound_students - number_of_separated_students + 1
  let gaps := remaining_elements + 1
  bound_ways * arrangements remaining_elements remaining_elements * arrangements gaps number_of_separated_students = 960 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l3847_384740


namespace NUMINAMATH_CALUDE_divisible_by_13_with_sqrt_between_24_and_24_5_verify_585_and_598_l3847_384761

theorem divisible_by_13_with_sqrt_between_24_and_24_5 : 
  ∃ (n : ℕ), n > 0 ∧ n % 13 = 0 ∧ 24 < Real.sqrt n ∧ Real.sqrt n < 24.5 :=
by
  sorry

theorem verify_585_and_598 : 
  (585 > 0 ∧ 585 % 13 = 0 ∧ 24 < Real.sqrt 585 ∧ Real.sqrt 585 < 24.5) ∧
  (598 > 0 ∧ 598 % 13 = 0 ∧ 24 < Real.sqrt 598 ∧ Real.sqrt 598 < 24.5) :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_13_with_sqrt_between_24_and_24_5_verify_585_and_598_l3847_384761


namespace NUMINAMATH_CALUDE_range_of_c_l3847_384751

-- Define the propositions P and Q
def P (c : ℝ) : Prop := ∀ x : ℝ, Monotone (fun x => (c^2 - 5*c + 7)^x)

def Q (c : ℝ) : Prop := ∀ x : ℝ, |x - 1| + |x - 2*c| > 1

-- Define the theorem
theorem range_of_c :
  (∃! c : ℝ, P c ∨ Q c) →
  {c : ℝ | c ∈ Set.Icc 0 1 ∪ Set.Icc 2 3} = {c : ℝ | P c ∨ Q c} :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l3847_384751


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3847_384724

theorem quadratic_inequality_range :
  ∃ a : ℝ, a ∈ Set.Icc 1 3 ∧ ∀ x : ℝ, a * x^2 + (a - 2) * x - 2 > 0 →
    x < -1 ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3847_384724


namespace NUMINAMATH_CALUDE_newspaper_cost_theorem_l3847_384733

/-- The cost of a weekday newspaper -/
def weekday_cost : ℚ := 1/2

/-- The cost of a Sunday newspaper -/
def sunday_cost : ℚ := 2

/-- The number of weekday newspapers bought per week -/
def weekday_papers_per_week : ℕ := 3

/-- The number of weeks -/
def num_weeks : ℕ := 8

/-- The total cost of newspapers over the given number of weeks -/
def total_cost : ℚ := num_weeks * (weekday_papers_per_week * weekday_cost + sunday_cost)

theorem newspaper_cost_theorem : total_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_theorem_l3847_384733


namespace NUMINAMATH_CALUDE_equation_solution_l3847_384749

theorem equation_solution (m n : ℚ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-2) = 6) → 
  m = 4.5 ∧ n = 1.5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3847_384749


namespace NUMINAMATH_CALUDE_bakery_sales_theorem_l3847_384797

/-- Represents the bakery sales scenario -/
structure BakerySales where
  pumpkin_slices_per_pie : ℕ
  custard_slices_per_pie : ℕ
  pumpkin_price_per_slice : ℕ
  custard_price_per_slice : ℕ
  pumpkin_pies_sold : ℕ
  custard_pies_sold : ℕ

/-- Calculates the total sales from the bakery -/
def total_sales (s : BakerySales) : ℕ :=
  (s.pumpkin_slices_per_pie * s.pumpkin_pies_sold * s.pumpkin_price_per_slice) +
  (s.custard_slices_per_pie * s.custard_pies_sold * s.custard_price_per_slice)

/-- Theorem stating that given the specific conditions, the total sales equal $340 -/
theorem bakery_sales_theorem (s : BakerySales) 
  (h1 : s.pumpkin_slices_per_pie = 8)
  (h2 : s.custard_slices_per_pie = 6)
  (h3 : s.pumpkin_price_per_slice = 5)
  (h4 : s.custard_price_per_slice = 6)
  (h5 : s.pumpkin_pies_sold = 4)
  (h6 : s.custard_pies_sold = 5) :
  total_sales s = 340 := by
  sorry

#eval total_sales {
  pumpkin_slices_per_pie := 8,
  custard_slices_per_pie := 6,
  pumpkin_price_per_slice := 5,
  custard_price_per_slice := 6,
  pumpkin_pies_sold := 4,
  custard_pies_sold := 5
}

end NUMINAMATH_CALUDE_bakery_sales_theorem_l3847_384797


namespace NUMINAMATH_CALUDE_tennis_players_count_l3847_384721

theorem tennis_players_count (total : ℕ) (badminton : ℕ) (both : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 18 →
  both = 9 →
  neither = 2 →
  ∃ tennis : ℕ, tennis = 19 ∧ 
    total = badminton + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_count_l3847_384721


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l3847_384762

/-- Given two lines l₁ and l in the plane, this theorem states that the line l₂ 
    which is symmetric to l₁ with respect to l has a specific equation. -/
theorem symmetric_line_equation (x y : ℝ) : 
  let l₁ : ℝ → ℝ := λ x => 2 * x
  let l : ℝ → ℝ := λ x => 3 * x + 3
  let l₂ : ℝ → ℝ := λ x => (11 * x - 21) / 2
  (∀ x, l₂ x = y ↔ 11 * x - 2 * y + 21 = 0) ∧
  (∀ p : ℝ × ℝ, 
    let p₁ := (p.1, l₁ p.1)
    let m := ((p.1 + p₁.1) / 2, (p.2 + p₁.2) / 2)
    m.2 = l m.1 → p.2 = l₂ p.1) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_line_equation_l3847_384762


namespace NUMINAMATH_CALUDE_calculate_expression_l3847_384705

theorem calculate_expression : |(-8 : ℝ)| + (-2011 : ℝ)^0 - 2 * Real.cos (π / 3) + (1 / 2)⁻¹ = 10 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3847_384705


namespace NUMINAMATH_CALUDE_least_skilled_painter_is_granddaughter_l3847_384719

-- Define the family members
inductive FamilyMember
  | Grandmother
  | Niece
  | Nephew
  | Granddaughter

-- Define the skill levels
inductive SkillLevel
  | Best
  | Least

-- Define the gender
inductive Gender
  | Male
  | Female

-- Function to get the gender of a family member
def gender (m : FamilyMember) : Gender :=
  match m with
  | FamilyMember.Grandmother => Gender.Female
  | FamilyMember.Niece => Gender.Female
  | FamilyMember.Nephew => Gender.Male
  | FamilyMember.Granddaughter => Gender.Female

-- Function to determine if two family members can be twins
def canBeTwins (m1 m2 : FamilyMember) : Prop :=
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Niece) ∨
  (m1 = FamilyMember.Granddaughter ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Granddaughter)

-- Function to determine if two family members can be the same age
def canBeSameAge (m1 m2 : FamilyMember) : Prop :=
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Nephew) ∨
  (m1 = FamilyMember.Nephew ∧ m2 = FamilyMember.Niece) ∨
  (m1 = FamilyMember.Niece ∧ m2 = FamilyMember.Granddaughter) ∨
  (m1 = FamilyMember.Granddaughter ∧ m2 = FamilyMember.Niece)

-- Theorem statement
theorem least_skilled_painter_is_granddaughter :
  ∀ (best least : FamilyMember),
    (gender best ≠ gender least) →
    (∃ twin, canBeTwins twin least ∧ twin ≠ least) →
    canBeSameAge best least →
    least = FamilyMember.Granddaughter :=
by
  sorry

end NUMINAMATH_CALUDE_least_skilled_painter_is_granddaughter_l3847_384719


namespace NUMINAMATH_CALUDE_jafari_candy_count_l3847_384757

theorem jafari_candy_count (total candy_taquon candy_mack : ℕ) 
  (h1 : total = candy_taquon + candy_mack + (total - candy_taquon - candy_mack))
  (h2 : candy_taquon = 171)
  (h3 : candy_mack = 171)
  (h4 : total = 418) :
  total - candy_taquon - candy_mack = 76 := by
sorry

end NUMINAMATH_CALUDE_jafari_candy_count_l3847_384757


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3847_384743

/-- Given a right-angled triangle with hypotenuse 5000 km and one other side 4000 km,
    the sum of all sides is 12000 km. -/
theorem triangle_perimeter (a b c : ℝ) (h1 : a = 5000) (h2 : b = 4000) 
    (h3 : a^2 = b^2 + c^2) : a + b + c = 12000 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3847_384743


namespace NUMINAMATH_CALUDE_impossibleEvent_l3847_384718

/-- A fair dice with faces numbered 1 to 6 -/
def Dice : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The event of getting a number divisible by 10 when rolling the dice -/
def DivisibleBy10 (n : ℕ) : Prop := n % 10 = 0

/-- Theorem: The event of rolling a number divisible by 10 is impossible -/
theorem impossibleEvent : ∀ n ∈ Dice, ¬ DivisibleBy10 n := by
  sorry

end NUMINAMATH_CALUDE_impossibleEvent_l3847_384718


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l3847_384776

theorem triangle_cosine_sum (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  b * Real.cos C + c * Real.cos B = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l3847_384776


namespace NUMINAMATH_CALUDE_vector_equality_l3847_384774

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, and D in a vector space,
    DA + CD - CB = BA -/
theorem vector_equality (A B C D : V) : D - A + (C - D) - (C - B) = B - A := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l3847_384774


namespace NUMINAMATH_CALUDE_modified_square_boundary_length_l3847_384712

/-- The boundary length of a modified square figure --/
theorem modified_square_boundary_length :
  ∀ (square_area : ℝ) (num_segments : ℕ),
    square_area = 100 →
    num_segments = 4 →
    ∃ (boundary_length : ℝ),
      boundary_length = 5 * Real.pi + 10 := by
  sorry

end NUMINAMATH_CALUDE_modified_square_boundary_length_l3847_384712


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3847_384734

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 6}
def B : Set ℤ := {1, 4, 5}

theorem intersection_complement_equals : A ∩ (U \ B) = {3, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3847_384734


namespace NUMINAMATH_CALUDE_root_product_equation_l3847_384736

theorem root_product_equation (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b^2)^2 - p*(a^2 + 1/b^2) + r = 0) →
  ((b^2 + 1/a^2)^2 - p*(b^2 + 1/a^2) + r = 0) →
  r = 100/9 := by
sorry

end NUMINAMATH_CALUDE_root_product_equation_l3847_384736


namespace NUMINAMATH_CALUDE_line_slope_is_two_l3847_384779

/-- Given a line ax + y - 4 = 0 passing through the point (-1, 2), prove that its slope is 2 -/
theorem line_slope_is_two (a : ℝ) : 
  (a * (-1) + 2 - 4 = 0) → -- Line passes through (-1, 2)
  (∃ m b : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = m * x + b) → -- Line can be written in slope-intercept form
  (∃ m : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = m * x + 4) → -- Specific y-intercept is 4
  (∃ m : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = 2 * x + 4) -- Slope is 2
  := by sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l3847_384779


namespace NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l3847_384785

theorem least_integer_square_72_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 72 ∧ ∀ y : ℤ, y^2 = 2*y + 72 → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l3847_384785


namespace NUMINAMATH_CALUDE_no_all_permutations_perfect_squares_l3847_384700

/-- A function that checks if a natural number has all non-zero digits -/
def allDigitsNonZero (n : ℕ) : Prop := sorry

/-- A function that generates all permutations of digits of a natural number -/
def digitPermutations (n : ℕ) : Set ℕ := sorry

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := sorry

theorem no_all_permutations_perfect_squares :
  ∀ n : ℕ, n ≥ 10 → allDigitsNonZero n →
    ∃ m ∈ digitPermutations n, ¬ isPerfectSquare m :=
sorry

end NUMINAMATH_CALUDE_no_all_permutations_perfect_squares_l3847_384700


namespace NUMINAMATH_CALUDE_base6_addition_correct_l3847_384787

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The first number in base 6 -/
def num1 : List Nat := [3, 4, 2, 1]

/-- The second number in base 6 -/
def num2 : List Nat := [4, 5, 2, 5]

/-- The expected sum in base 6 -/
def expectedSum : List Nat := [1, 2, 3, 5, 0]

theorem base6_addition_correct :
  decimalToBase6 (base6ToDecimal num1 + base6ToDecimal num2) = expectedSum := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_correct_l3847_384787


namespace NUMINAMATH_CALUDE_solution_value_l3847_384728

theorem solution_value (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3847_384728


namespace NUMINAMATH_CALUDE_exactly_three_rainy_days_l3847_384710

/-- The probability of exactly k successes in n independent trials
    with probability p of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day -/
def rain_probability : ℝ := 0.5

/-- The number of days considered -/
def num_days : ℕ := 4

/-- The number of rainy days we're interested in -/
def num_rainy_days : ℕ := 3

theorem exactly_three_rainy_days :
  binomial_probability num_days num_rainy_days rain_probability = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_rainy_days_l3847_384710


namespace NUMINAMATH_CALUDE_unique_grouping_l3847_384764

def numbers : List ℕ := [12, 30, 42, 44, 57, 91, 95, 143]

def is_valid_grouping (group1 group2 : List ℕ) : Prop :=
  group1.prod = group2.prod ∧
  (group1 ++ group2).toFinset = numbers.toFinset ∧
  group1.toFinset ∩ group2.toFinset = ∅

theorem unique_grouping :
  ∀ (group1 group2 : List ℕ),
    is_valid_grouping group1 group2 →
    ((group1.toFinset = {12, 42, 95, 143} ∧ group2.toFinset = {30, 44, 57, 91}) ∨
     (group2.toFinset = {12, 42, 95, 143} ∧ group1.toFinset = {30, 44, 57, 91})) :=
by sorry

end NUMINAMATH_CALUDE_unique_grouping_l3847_384764


namespace NUMINAMATH_CALUDE_sunshine_cost_per_mile_is_correct_l3847_384723

/-- The cost per mile for Sunshine Car Rentals -/
def sunshine_cost_per_mile : ℝ := 0.18

/-- The daily rate for Sunshine Car Rentals -/
def sunshine_daily_rate : ℝ := 17.99

/-- The daily rate for City Rentals -/
def city_daily_rate : ℝ := 18.95

/-- The cost per mile for City Rentals -/
def city_cost_per_mile : ℝ := 0.16

/-- The number of miles at which the costs are equal -/
def equal_cost_miles : ℝ := 48.0

theorem sunshine_cost_per_mile_is_correct :
  sunshine_daily_rate + equal_cost_miles * sunshine_cost_per_mile =
  city_daily_rate + equal_cost_miles * city_cost_per_mile :=
by sorry

end NUMINAMATH_CALUDE_sunshine_cost_per_mile_is_correct_l3847_384723


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3847_384702

-- Define the quadratic function
def quadratic_function (m x : ℝ) : ℝ := m * x^2 - 4 * x + 1

-- State the theorem
theorem quadratic_minimum_value (m : ℝ) :
  (∃ x_min : ℝ, ∀ x : ℝ, quadratic_function m x ≥ quadratic_function m x_min) ∧
  (∃ x_min : ℝ, quadratic_function m x_min = -3) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3847_384702


namespace NUMINAMATH_CALUDE_garden_fencing_l3847_384704

theorem garden_fencing (garden_area : ℝ) (extension : ℝ) : 
  garden_area = 784 →
  extension = 10 →
  (4 * (Real.sqrt garden_area + extension)) = 152 := by
  sorry

end NUMINAMATH_CALUDE_garden_fencing_l3847_384704


namespace NUMINAMATH_CALUDE_fraction_seven_twentynine_repetend_l3847_384772

/-- The repetend of a rational number is the repeating part of its decimal representation. -/
def repetend (n d : ℕ) : ℕ := sorry

/-- A number is a valid repetend for a fraction if it repeats infinitely in the decimal representation. -/
def is_valid_repetend (r n d : ℕ) : Prop := sorry

theorem fraction_seven_twentynine_repetend :
  let r := 241379
  is_valid_repetend r 7 29 ∧ repetend 7 29 = r :=
sorry

end NUMINAMATH_CALUDE_fraction_seven_twentynine_repetend_l3847_384772


namespace NUMINAMATH_CALUDE_double_y_plus_8_not_less_than_negative_3_l3847_384794

theorem double_y_plus_8_not_less_than_negative_3 :
  ∀ y : ℝ, (2 * y + 8 ≥ -3) ↔ (∃ z : ℝ, z = 2 * y ∧ z + 8 ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_double_y_plus_8_not_less_than_negative_3_l3847_384794


namespace NUMINAMATH_CALUDE_jar_weight_percentage_l3847_384720

theorem jar_weight_percentage (jar_weight : ℝ) (full_weight : ℝ) 
  (h1 : jar_weight = 0.4 * full_weight)
  (h2 : jar_weight > 0)
  (h3 : full_weight > jar_weight) :
  let beans_weight := full_weight - jar_weight
  let remaining_beans_weight := (1/3) * beans_weight
  let new_total_weight := jar_weight + remaining_beans_weight
  new_total_weight / full_weight = 0.6 := by
sorry

end NUMINAMATH_CALUDE_jar_weight_percentage_l3847_384720


namespace NUMINAMATH_CALUDE_second_number_in_list_l3847_384799

theorem second_number_in_list (n x : ℤ) : 
  (3 + n + 138 + 1917 + 2114 + x) / 6 = 12 →
  x % 7 = 0 →
  n = 915 :=
by sorry

end NUMINAMATH_CALUDE_second_number_in_list_l3847_384799


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l3847_384798

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The set of digits used to form the numbers. -/
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The number of digits in each formed number. -/
def number_length : ℕ := 7

/-- The digit that must be at the last position. -/
def last_digit : ℕ := 3

/-- The theorem stating the number of valid arrangements. -/
theorem count_valid_arrangements :
  (factorial (number_length - 1) / 2 : ℕ) = 360 := by sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l3847_384798


namespace NUMINAMATH_CALUDE_seven_strip_trapezoid_shaded_area_l3847_384722

/-- Represents a trapezoid divided into equal width strips -/
structure StripTrapezoid where
  numStrips : ℕ
  numShaded : ℕ
  h_pos : 0 < numStrips
  h_shaded : numShaded ≤ numStrips

/-- The fraction of shaded area in a strip trapezoid -/
def shadedAreaFraction (t : StripTrapezoid) : ℚ :=
  t.numShaded / t.numStrips

/-- Theorem: In a trapezoid divided into 7 strips with 4 shaded, the shaded area is 4/7 of the total area -/
theorem seven_strip_trapezoid_shaded_area :
  let t : StripTrapezoid := ⟨7, 4, by norm_num, by norm_num⟩
  shadedAreaFraction t = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_strip_trapezoid_shaded_area_l3847_384722


namespace NUMINAMATH_CALUDE_children_at_track_meet_l3847_384732

theorem children_at_track_meet (total_seats : ℕ) (empty_seats : ℕ) (adults : ℕ) 
  (h1 : total_seats = 95)
  (h2 : empty_seats = 14)
  (h3 : adults = 29) :
  total_seats - empty_seats - adults = 52 := by
  sorry

end NUMINAMATH_CALUDE_children_at_track_meet_l3847_384732


namespace NUMINAMATH_CALUDE_actual_daily_length_is_72_required_daily_increase_at_least_36_l3847_384739

/-- Represents the renovation of a pipe network --/
structure PipeRenovation where
  totalLength : ℝ
  originalDailyLength : ℝ
  efficiencyIncrease : ℝ
  daysAheadOfSchedule : ℝ
  constructedDays : ℝ
  maxTotalDays : ℝ

/-- Calculates the actual daily renovation length --/
def actualDailyLength (pr : PipeRenovation) : ℝ :=
  pr.originalDailyLength * (1 + pr.efficiencyIncrease)

/-- Theorem for the actual daily renovation length --/
theorem actual_daily_length_is_72 (pr : PipeRenovation)
  (h1 : pr.totalLength = 3600)
  (h2 : pr.efficiencyIncrease = 0.2)
  (h3 : pr.daysAheadOfSchedule = 10)
  (h4 : pr.totalLength / pr.originalDailyLength - pr.totalLength / (actualDailyLength pr) = pr.daysAheadOfSchedule) :
  actualDailyLength pr = 72 := by sorry

/-- Theorem for the required increase in daily renovation length --/
theorem required_daily_increase_at_least_36 (pr : PipeRenovation)
  (h1 : pr.totalLength = 3600)
  (h2 : actualDailyLength pr = 72)
  (h3 : pr.constructedDays = 20)
  (h4 : pr.maxTotalDays = 40) :
  ∃ m : ℝ, m ≥ 36 ∧ (pr.maxTotalDays - pr.constructedDays) * (actualDailyLength pr + m) ≥ pr.totalLength - actualDailyLength pr * pr.constructedDays := by sorry

end NUMINAMATH_CALUDE_actual_daily_length_is_72_required_daily_increase_at_least_36_l3847_384739


namespace NUMINAMATH_CALUDE_common_difference_proof_l3847_384706

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_proof (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h2 : a 2 = 14) (h5 : a 5 = 5) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -3 := by
sorry

end NUMINAMATH_CALUDE_common_difference_proof_l3847_384706


namespace NUMINAMATH_CALUDE_area_equality_iff_rectangle_l3847_384781

/-- A quadrilateral with sides a, b, c, d and area A -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  A : ℝ

/-- Definition of a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∃ (w h : ℝ), q.a = w ∧ q.b = h ∧ q.c = w ∧ q.d = h ∧ q.A = w * h

/-- Theorem: Area equality holds iff the quadrilateral is a rectangle -/
theorem area_equality_iff_rectangle (q : Quadrilateral) :
  q.A = ((q.a + q.c) / 2) * ((q.b + q.d) / 2) ↔ is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_area_equality_iff_rectangle_l3847_384781


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3847_384791

theorem stratified_sample_size 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_female : ℕ) 
  (h1 : total_male = 42) 
  (h2 : total_female = 30) 
  (h3 : sample_female = 5) :
  ∃ (sample_male : ℕ), 
    (sample_male : ℚ) / (sample_female : ℚ) = (total_male : ℚ) / (total_female : ℚ) ∧
    sample_male + sample_female = 12 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3847_384791


namespace NUMINAMATH_CALUDE_reflect_M_y_axis_l3847_384768

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point M -/
def M : ℝ × ℝ := (3, 2)

/-- Theorem: Reflecting M(3,2) across the y-axis results in (-3,2) -/
theorem reflect_M_y_axis : reflect_y M = (-3, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_M_y_axis_l3847_384768


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3847_384790

/-- The axis of symmetry for the parabola x = -4y² is x = 1/16 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := fun y ↦ -4 * y^2
  ∃ x₀ : ℝ, x₀ = 1/16 ∧ ∀ y : ℝ, f y = f (-y) → x₀ = f y :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3847_384790


namespace NUMINAMATH_CALUDE_adams_trivia_score_l3847_384750

/-- Adam's trivia game score calculation -/
theorem adams_trivia_score :
  ∀ (first_half second_half points_per_question : ℕ),
  first_half = 8 →
  second_half = 2 →
  points_per_question = 8 →
  (first_half + second_half) * points_per_question = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_adams_trivia_score_l3847_384750


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3847_384744

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 2) = -26 + k * x) ↔ 
  (k = -2 + 6 * Real.sqrt 6 ∨ k = -2 - 6 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3847_384744


namespace NUMINAMATH_CALUDE_first_replaced_man_age_l3847_384715

/-- The age of the first replaced man in a group scenario --/
def age_of_first_replaced_man (initial_count : ℕ) (age_increase : ℕ) (second_replaced_age : ℕ) (new_men_average_age : ℕ) : ℕ :=
  initial_count * age_increase + new_men_average_age * 2 - second_replaced_age - initial_count * age_increase

/-- Theorem stating the age of the first replaced man is 21 --/
theorem first_replaced_man_age :
  age_of_first_replaced_man 15 2 23 37 = 21 := by
  sorry

#eval age_of_first_replaced_man 15 2 23 37

end NUMINAMATH_CALUDE_first_replaced_man_age_l3847_384715


namespace NUMINAMATH_CALUDE_min_difference_of_bounds_l3847_384760

-- Define the arithmetic-geometric sequence
def a (n : ℕ) : ℚ := (4/3) * (-1/3)^(n-1)

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := 1 - (-1/3)^n

-- Define the function f(n) = S(n) - 1/S(n)
def f (n : ℕ) : ℚ := S n - 1 / (S n)

-- Theorem statement
theorem min_difference_of_bounds (A B : ℚ) :
  (∀ n : ℕ, n ≥ 1 → A ≤ f n ∧ f n ≤ B) →
  B - A ≥ 59/72 :=
sorry

end NUMINAMATH_CALUDE_min_difference_of_bounds_l3847_384760


namespace NUMINAMATH_CALUDE_triangle_side_length_l3847_384792

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →   -- Area of the triangle is √3
  (B = π/3) →                                 -- Angle B is 60°
  (a^2 + c^2 = 3*a*c) →                        -- Given condition
  (b = 2 * Real.sqrt 2) :=                     -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3847_384792


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l3847_384758

theorem quadratic_equation_sum (x p q : ℝ) : 
  (5 * x^2 - 30 * x - 45 = 0) → 
  ((x + p)^2 = q) → 
  (p + q = 15) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l3847_384758


namespace NUMINAMATH_CALUDE_exists_number_satisfying_equation_l3847_384709

theorem exists_number_satisfying_equation : ∃ x : ℝ, (3.241 * x) / 100 = 0.045374000000000005 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_equation_l3847_384709


namespace NUMINAMATH_CALUDE_cousins_distribution_l3847_384730

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
theorem cousins_distribution : distribute 5 4 = 66 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l3847_384730


namespace NUMINAMATH_CALUDE_hong_travel_bound_l3847_384767

/-- Represents a town in the country -/
structure Town where
  coins : ℕ

/-- Represents the country with its towns and roads -/
structure Country where
  towns : Set Town
  roads : Set (Town × Town)
  initial_coins : ℕ

/-- Represents Hong's travel -/
structure Travel where
  country : Country
  days : ℕ

/-- The maximum number of days Hong can travel -/
def max_travel_days (n : ℕ) : ℕ := n + 2 * n^(2/3)

theorem hong_travel_bound (c : Country) (t : Travel) (h_infinite : Infinite c.towns)
    (h_all_connected : ∀ a b : Town, a ≠ b → (a, b) ∈ c.roads)
    (h_initial_coins : ∀ town ∈ c.towns, town.coins = c.initial_coins)
    (h_coin_transfer : ∀ k : ℕ, ∀ a b : Town, 
      (a, b) ∈ c.roads → t.days = k → b.coins = b.coins - k ∧ a.coins = a.coins + k)
    (h_road_usage : ∀ a b : Town, (a, b) ∈ c.roads → (b, a) ∉ c.roads) :
  t.days ≤ max_travel_days c.initial_coins :=
sorry

end NUMINAMATH_CALUDE_hong_travel_bound_l3847_384767


namespace NUMINAMATH_CALUDE_power_of_256_l3847_384786

theorem power_of_256 : (256 : ℝ) ^ (5/8 : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_of_256_l3847_384786


namespace NUMINAMATH_CALUDE_inequality_proof_l3847_384777

theorem inequality_proof (a : ℝ) : 3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3847_384777


namespace NUMINAMATH_CALUDE_smallest_4digit_base7_divisible_by_7_l3847_384771

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 7 number --/
def is4DigitBase7 (n : ℕ) : Prop := sorry

/-- The smallest 4-digit base 7 number --/
def smallestBase7_4Digit : ℕ := 1000

theorem smallest_4digit_base7_divisible_by_7 :
  (is4DigitBase7 smallestBase7_4Digit) ∧
  (base7ToDecimal smallestBase7_4Digit % 7 = 0) ∧
  (∀ n : ℕ, is4DigitBase7 n ∧ n < smallestBase7_4Digit → base7ToDecimal n % 7 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_4digit_base7_divisible_by_7_l3847_384771


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3847_384755

theorem arithmetic_calculations :
  ((54 + 38) * 15 = 1380) ∧
  (1500 - 32 * 45 = 60) ∧
  (157 * (70 / 35) = 314) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3847_384755


namespace NUMINAMATH_CALUDE_smallest_with_144_divisors_and_10_consecutive_l3847_384717

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has 10 consecutive divisors -/
def has_10_consecutive_divisors (n : ℕ) : Prop := sorry

/-- The theorem stating that 110880 is the smallest number satisfying the conditions -/
theorem smallest_with_144_divisors_and_10_consecutive : 
  num_divisors 110880 = 144 ∧ 
  has_10_consecutive_divisors 110880 ∧ 
  ∀ m : ℕ, m < 110880 → (num_divisors m ≠ 144 ∨ ¬has_10_consecutive_divisors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_144_divisors_and_10_consecutive_l3847_384717


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_power_minus_75_l3847_384780

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_large_power_minus_75 :
  sum_of_digits (10^50 - 75) = 439 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_power_minus_75_l3847_384780


namespace NUMINAMATH_CALUDE_fraction_sum_and_multiply_l3847_384703

theorem fraction_sum_and_multiply :
  3 * (2 / 10 + 4 / 20 + 6 / 30) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_multiply_l3847_384703


namespace NUMINAMATH_CALUDE_partnership_profit_l3847_384711

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ  -- A's investment
  b_investment : ℕ  -- B's investment
  a_period : ℕ     -- A's investment period
  b_period : ℕ     -- B's investment period
  b_profit : ℕ     -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℕ :=
  let a_profit := p.b_profit * 6
  a_profit + p.b_profit

/-- Theorem stating the total profit for the given partnership conditions --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_investment = 3 * p.b_investment)
  (h2 : p.a_period = 2 * p.b_period)
  (h3 : p.b_profit = 4500) : 
  total_profit p = 31500 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 4500 }

end NUMINAMATH_CALUDE_partnership_profit_l3847_384711


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a4_l3847_384773

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem arithmetic_geometric_sequence_a4
  (a : ℕ → ℝ)
  (h_seq : ArithmeticGeometricSequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_a4_l3847_384773


namespace NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l3847_384745

theorem power_of_seven_mod_thousand : 7^2023 % 1000 = 637 := by sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l3847_384745


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l3847_384714

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 1}

-- Theorem for A ⊆ B
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 1/2 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 3/2 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l3847_384714


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3847_384783

theorem fraction_sum_equality : (18 : ℚ) / 45 - 2 / 9 + 1 / 6 = 31 / 90 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3847_384783


namespace NUMINAMATH_CALUDE_hex_A08_equals_2568_l3847_384713

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  if c.isDigit then c.toNat - '0'.toNat
  else if 'A' ≤ c ∧ c ≤ 'F' then c.toNat - 'A'.toNat + 10
  else 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal representation of the number -/
def hex_number : String := "A08"

/-- Theorem stating that the hexadecimal number A08 is equal to 2568 in decimal -/
theorem hex_A08_equals_2568 : hex_string_to_dec hex_number = 2568 := by
  sorry


end NUMINAMATH_CALUDE_hex_A08_equals_2568_l3847_384713


namespace NUMINAMATH_CALUDE_sons_age_l3847_384725

/-- Proves that given the conditions, the son's age is 26 years -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry


end NUMINAMATH_CALUDE_sons_age_l3847_384725


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3847_384793

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (69/29, 43/29)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 5*x - 6*y = 3

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 8*x + 2*y = 22

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3847_384793


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l3847_384738

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l3847_384738


namespace NUMINAMATH_CALUDE_carnival_tickets_billy_carnival_tickets_l3847_384770

/-- Calculate the total number of tickets used at a carnival --/
theorem carnival_tickets (ferris_wheel_rides bumper_car_rides ferris_wheel_cost bumper_car_cost : ℕ) :
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost =
  (ferris_wheel_rides * ferris_wheel_cost) + (bumper_car_rides * bumper_car_cost) := by
  sorry

/-- Billy's carnival ticket usage --/
theorem billy_carnival_tickets :
  let ferris_wheel_rides : ℕ := 7
  let bumper_car_rides : ℕ := 3
  let ferris_wheel_cost : ℕ := 6
  let bumper_car_cost : ℕ := 4
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := by
  sorry

end NUMINAMATH_CALUDE_carnival_tickets_billy_carnival_tickets_l3847_384770


namespace NUMINAMATH_CALUDE_f_properties_l3847_384778

-- Define the function f
def f (x : ℝ) : ℝ := -x - x^3

-- State the theorem
theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3847_384778


namespace NUMINAMATH_CALUDE_worker_overtime_hours_l3847_384765

/-- A worker's pay calculation --/
theorem worker_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (overtime_rate : ℚ) (total_pay : ℚ) : 
  regular_rate = 3 →
  regular_hours = 40 →
  overtime_rate = 2 * regular_rate →
  total_pay = 180 →
  (total_pay - regular_rate * regular_hours) / overtime_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_worker_overtime_hours_l3847_384765


namespace NUMINAMATH_CALUDE_total_rainfall_is_correct_l3847_384707

-- Define conversion factors
def inch_to_cm : ℝ := 2.54
def mm_to_cm : ℝ := 0.1

-- Define daily rainfall measurements
def monday_rain : ℝ := 0.12962962962962962
def tuesday_rain : ℝ := 3.5185185185185186
def wednesday_rain : ℝ := 0.09259259259259259
def thursday_rain : ℝ := 0.10222222222222223
def friday_rain : ℝ := 12.222222222222221
def saturday_rain : ℝ := 0.2222222222222222
def sunday_rain : ℝ := 0.17444444444444446

-- Define the units for each day's measurement
inductive RainUnit
| Centimeter
| Millimeter
| Inch

def monday_unit : RainUnit := RainUnit.Centimeter
def tuesday_unit : RainUnit := RainUnit.Millimeter
def wednesday_unit : RainUnit := RainUnit.Centimeter
def thursday_unit : RainUnit := RainUnit.Inch
def friday_unit : RainUnit := RainUnit.Millimeter
def saturday_unit : RainUnit := RainUnit.Centimeter
def sunday_unit : RainUnit := RainUnit.Inch

-- Function to convert a measurement to centimeters based on its unit
def to_cm (measurement : ℝ) (unit : RainUnit) : ℝ :=
  match unit with
  | RainUnit.Centimeter => measurement
  | RainUnit.Millimeter => measurement * mm_to_cm
  | RainUnit.Inch => measurement * inch_to_cm

-- Theorem statement
theorem total_rainfall_is_correct : 
  to_cm monday_rain monday_unit +
  to_cm tuesday_rain tuesday_unit +
  to_cm wednesday_rain wednesday_unit +
  to_cm thursday_rain thursday_unit +
  to_cm friday_rain friday_unit +
  to_cm saturday_rain saturday_unit +
  to_cm sunday_rain sunday_unit = 2.721212629851652 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_is_correct_l3847_384707


namespace NUMINAMATH_CALUDE_solution_value_l3847_384763

theorem solution_value (m : ℝ) : (3 * m - 2 * 3 = 6) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3847_384763


namespace NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l3847_384742

theorem modulo_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 45689 ≡ n [ZMOD 23] ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l3847_384742


namespace NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l3847_384753

/-- Given a cylinder whose lateral surface unfolds to a square with side length 2,
    prove that its volume is 2/π. -/
theorem cylinder_volume_from_lateral_surface (r h : ℝ) : 
  (2 * π * r = 2) → (h = 2) → (π * r^2 * h = 2/π) := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_lateral_surface_l3847_384753
