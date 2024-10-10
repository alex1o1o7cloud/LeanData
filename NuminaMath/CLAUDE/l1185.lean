import Mathlib

namespace geometry_problem_l1185_118587

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the line 2x + y = 0
def line_center (x y : ℝ) : Prop := 2*x + y = 0

theorem geometry_problem :
  -- Conditions
  (∀ x y, line_l x y → (x = 2 ∧ y = -1) → True) ∧  -- l passes through P(2,-1)
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, line_l x y ↔ x/a + y/b = 1) ∧  -- Sum of intercepts is 2
  (∃ m, line_center m (-2*m) ∧ ∀ x y, circle_M x y → line_center x y) ∧  -- M's center on 2x+y=0
  (∀ x y, circle_M x y → line_l x y → (x = 2 ∧ y = -1)) →  -- M tangent to l at P
  -- Conclusions
  (∀ x y, line_l x y ↔ x + y = 1) ∧  -- Equation of line l
  (∀ x y, circle_M x y ↔ (x - 1)^2 + (y + 2)^2 = 2) ∧  -- Equation of circle M
  (∃ y₁ y₂, y₁ < y₂ ∧ circle_M 0 y₁ ∧ circle_M 0 y₂ ∧ y₂ - y₁ = 2)  -- Length of chord on y-axis
  := by sorry

end geometry_problem_l1185_118587


namespace product_expansion_sum_l1185_118581

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2 * x^2 - 3 * x + 5) * (5 - x) = a * x^3 + b * x^2 + c * x + d) →
  27 * a + 9 * b + 3 * c + d = 28 := by
sorry

end product_expansion_sum_l1185_118581


namespace froglet_is_sane_l1185_118507

-- Define the servants
inductive Servant
| LackeyLecc
| Froglet

-- Define the sanity state
inductive SanityState
| Sane
| Insane

-- Define a function to represent the claim of Lackey-Lecc
def lackey_lecc_claim (lackey_state froglet_state : SanityState) : Prop :=
  (lackey_state = SanityState.Sane ∧ froglet_state = SanityState.Sane) ∨
  (lackey_state = SanityState.Insane ∧ froglet_state = SanityState.Insane)

-- Theorem stating that Froglet is sane
theorem froglet_is_sane :
  ∀ (lackey_state : SanityState),
    (lackey_lecc_claim lackey_state SanityState.Sane) →
    SanityState.Sane = SanityState.Sane :=
by
  sorry


end froglet_is_sane_l1185_118507


namespace length_AD_is_zero_l1185_118580

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ca := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  ab = 9 ∧ bc = 40 ∧ ca = 41

-- Define right angle at C
def RightAngleC (A B C : ℝ × ℝ) : Prop :=
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0

-- Define the circumscribed circle ω
def CircumscribedCircle (ω : Set (ℝ × ℝ)) (A B C : ℝ × ℝ) : Prop :=
  ∀ P : ℝ × ℝ, P ∈ ω ↔ (P.1 - A.1)^2 + (P.2 - A.2)^2 = 
                      (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧
                      (P.1 - B.1)^2 + (P.2 - B.2)^2 = 
                      (P.1 - C.1)^2 + (P.2 - C.2)^2

-- Define point D
def PointD (D : ℝ × ℝ) (ω : Set (ℝ × ℝ)) (A C : ℝ × ℝ) : Prop :=
  D ∈ ω ∧ 
  (D.1 - (A.1 + C.1)/2) * (C.2 - A.2) = (D.2 - (A.2 + C.2)/2) * (C.1 - A.1) ∧
  (D.1 - A.1) * (C.1 - A.1) + (D.2 - A.2) * (C.2 - A.2) < 0

theorem length_AD_is_zero 
  (A B C D : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : 
  Triangle A B C → 
  RightAngleC A B C → 
  CircumscribedCircle ω A B C → 
  PointD D ω A C → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 0 := by
    sorry

end length_AD_is_zero_l1185_118580


namespace vacant_seats_l1185_118573

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 75 / 100) : 
  (1 - filled_percentage) * total_seats = 150 := by
sorry


end vacant_seats_l1185_118573


namespace distance_to_centroid_l1185_118556

-- Define a triangle by its side lengths
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define a point inside the triangle by its distances from the vertices
structure InnerPoint (t : Triangle) where
  p : ℝ
  q : ℝ
  r : ℝ
  pos_p : 0 < p
  pos_q : 0 < q
  pos_r : 0 < r

-- Theorem statement
theorem distance_to_centroid (t : Triangle) (d : InnerPoint t) :
  ∃ (ds : ℝ), ds^2 = (3 * (d.p^2 + d.q^2 + d.r^2) - (t.a^2 + t.b^2 + t.c^2)) / 9 :=
sorry

end distance_to_centroid_l1185_118556


namespace intersection_A_complement_B_intersection_A_B_nonempty_iff_l1185_118597

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 1) = {x | 1 < x ∧ x < 3} := by sorry

-- Part 2
theorem intersection_A_B_nonempty_iff (k : ℝ) :
  (A ∩ B k).Nonempty ↔ k ≥ -1 := by sorry

end intersection_A_complement_B_intersection_A_B_nonempty_iff_l1185_118597


namespace competition_result_count_l1185_118554

/-- Represents a team's score composition -/
structure TeamScore where
  threes : ℕ  -- number of 3-point problems solved
  fives  : ℕ  -- number of 5-point problems solved

/-- Calculates the total score for a team -/
def totalScore (t : TeamScore) : ℕ := 3 * t.threes + 5 * t.fives

/-- Represents the scores of all three teams -/
structure CompetitionResult where
  team1 : TeamScore
  team2 : TeamScore
  team3 : TeamScore

/-- Checks if a competition result is valid -/
def isValidResult (r : CompetitionResult) : Prop :=
  totalScore r.team1 + totalScore r.team2 + totalScore r.team3 = 32

/-- Counts the number of valid competition results -/
def countValidResults : ℕ := sorry

theorem competition_result_count :
  countValidResults = 255 := by sorry

end competition_result_count_l1185_118554


namespace ratio_satisfies_conditions_l1185_118504

/-- Represents the number of people in each profession --/
structure ProfessionCount where
  doctors : ℕ
  lawyers : ℕ
  engineers : ℕ

/-- Checks if the given counts satisfy the average age conditions --/
def satisfiesAverageConditions (count : ProfessionCount) : Prop :=
  let totalPeople := count.doctors + count.lawyers + count.engineers
  let totalAge := 40 * count.doctors + 50 * count.lawyers + 60 * count.engineers
  totalAge / totalPeople = 45

/-- The theorem stating that the ratio 3:6:1 satisfies the conditions --/
theorem ratio_satisfies_conditions :
  ∃ (k : ℕ), k > 0 ∧ 
    let count : ProfessionCount := ⟨3*k, 6*k, k⟩
    satisfiesAverageConditions count :=
sorry

end ratio_satisfies_conditions_l1185_118504


namespace average_book_price_l1185_118530

theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 65 →
  books2 = 55 →
  price1 = 1280 →
  price2 = 880 →
  (price1 + price2) / (books1 + books2 : ℚ) = 18 := by
  sorry

end average_book_price_l1185_118530


namespace percentage_passed_all_subjects_l1185_118553

theorem percentage_passed_all_subjects 
  (fail_hindi : Real) 
  (fail_english : Real) 
  (fail_both : Real) 
  (fail_math : Real) 
  (h1 : fail_hindi = 0.2) 
  (h2 : fail_english = 0.7) 
  (h3 : fail_both = 0.1) 
  (h4 : fail_math = 0.5) : 
  (1 - (fail_hindi + fail_english - fail_both)) * (1 - fail_math) = 0.1 := by
  sorry

end percentage_passed_all_subjects_l1185_118553


namespace sum_a_c_l1185_118561

theorem sum_a_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 6) : 
  a + c = 7 := by sorry

end sum_a_c_l1185_118561


namespace valid_subset_of_A_l1185_118599

def A : Set ℝ := {x | x ≥ 0}

theorem valid_subset_of_A : 
  ({1, 2} : Set ℝ) ⊆ A ∧ 
  ¬({x : ℝ | x ≤ 1} ⊆ A) ∧ 
  ¬({-1, 0, 1} ⊆ A) ∧ 
  ¬(Set.univ ⊆ A) :=
sorry

end valid_subset_of_A_l1185_118599


namespace ab_max_and_sum_min_l1185_118558

theorem ab_max_and_sum_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 7 * b = 10) :
  (ab ≤ 25 / 21) ∧ (3 / a + 7 / b ≥ 10) := by
  sorry

end ab_max_and_sum_min_l1185_118558


namespace joan_sold_26_books_l1185_118501

/-- The number of books Joan sold in the yard sale -/
def books_sold (initial_books : ℕ) (remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

/-- Proof that Joan sold 26 books -/
theorem joan_sold_26_books (initial_books : ℕ) (remaining_books : ℕ) 
  (h1 : initial_books = 33) (h2 : remaining_books = 7) : 
  books_sold initial_books remaining_books = 26 := by
  sorry

#eval books_sold 33 7

end joan_sold_26_books_l1185_118501


namespace cat_gemstone_difference_l1185_118564

/-- Given three cats with gemstone collars, prove the difference between Spaatz's 
    gemstones and half of Frankie's gemstones. -/
theorem cat_gemstone_difference (binkie frankie spaatz : ℕ) : 
  binkie = 24 →
  spaatz = 1 →
  binkie = 4 * frankie →
  spaatz = frankie →
  spaatz - (frankie / 2) = 2 := by
  sorry

end cat_gemstone_difference_l1185_118564


namespace probability_at_least_one_vowel_l1185_118544

structure LetterSet where
  letters : Finset Char
  vowels : Finset Char
  vowels_subset : vowels ⊆ letters

def probability_no_vowel (s : LetterSet) : ℚ :=
  (s.letters.card - s.vowels.card : ℚ) / s.letters.card

def set1 : LetterSet := {
  letters := {'a', 'b', 'c', 'd', 'e'},
  vowels := {'a', 'e'},
  vowels_subset := by simp
}

def set2 : LetterSet := {
  letters := {'k', 'l', 'm', 'n', 'o', 'p'},
  vowels := ∅,
  vowels_subset := by simp
}

def set3 : LetterSet := {
  letters := {'r', 's', 't', 'u', 'v'},
  vowels := ∅,
  vowels_subset := by simp
}

def set4 : LetterSet := {
  letters := {'w', 'x', 'y', 'z', 'i'},
  vowels := {'i'},
  vowels_subset := by simp
}

theorem probability_at_least_one_vowel :
  1 - (probability_no_vowel set1 * probability_no_vowel set2 * 
       probability_no_vowel set3 * probability_no_vowel set4) = 17 / 20 := by
  sorry

end probability_at_least_one_vowel_l1185_118544


namespace barbara_savings_l1185_118576

/-- The number of weeks needed to save for a wristwatch -/
def weeks_to_save (watch_cost : ℕ) (weekly_allowance : ℕ) (current_savings : ℕ) : ℕ :=
  ((watch_cost - current_savings) + weekly_allowance - 1) / weekly_allowance

/-- Theorem: Given the conditions, Barbara needs 16 more weeks to save for the wristwatch -/
theorem barbara_savings : weeks_to_save 100 5 20 = 16 := by
  sorry

end barbara_savings_l1185_118576


namespace sequence_equality_l1185_118565

theorem sequence_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ (i : ℕ), a i = i := by
  sorry

end sequence_equality_l1185_118565


namespace addition_puzzle_l1185_118590

theorem addition_puzzle (A B C D : Nat) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  700 + 10 * A + 5 + 100 * B + 70 + C = 900 + 30 + 8 →
  D = 9 := by
sorry

end addition_puzzle_l1185_118590


namespace ellipse_equation_l1185_118535

/-- The standard equation of an ellipse with given major axis and eccentricity -/
theorem ellipse_equation (major_axis : ℝ) (eccentricity : ℝ) :
  major_axis = 8 ∧ eccentricity = 3/4 →
  ∃ (x y : ℝ), (x^2/16 + y^2/7 = 1) ∨ (x^2/7 + y^2/16 = 1) :=
by sorry

end ellipse_equation_l1185_118535


namespace sufficiency_not_necessity_l1185_118532

theorem sufficiency_not_necessity (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 → x₁ + x₂ > 2 ∧ x₁ * x₂ > 1) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ + x₂ > 2 ∧ x₁ * x₂ > 1 ∧ ¬(x₁ > 1 ∧ x₂ > 1)) :=
by sorry

end sufficiency_not_necessity_l1185_118532


namespace average_gas_mileage_l1185_118536

def total_distance : ℝ := 300
def sedan_efficiency : ℝ := 25
def truck_efficiency : ℝ := 15

theorem average_gas_mileage : 
  let sedan_distance := total_distance / 2
  let truck_distance := total_distance / 2
  let sedan_fuel := sedan_distance / sedan_efficiency
  let truck_fuel := truck_distance / truck_efficiency
  let total_fuel := sedan_fuel + truck_fuel
  (total_distance / total_fuel) = 18.75 := by sorry

end average_gas_mileage_l1185_118536


namespace sum_of_i_powers_l1185_118524

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the sum of specific powers of i equals i -/
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by sorry

end sum_of_i_powers_l1185_118524


namespace actual_speed_proof_l1185_118579

theorem actual_speed_proof (time_reduction : Real) (speed_increase : Real) 
  (h1 : time_reduction = Real.pi / 4)
  (h2 : speed_increase = Real.sqrt 15) : 
  ∃ (actual_speed : Real), actual_speed = Real.sqrt 15 := by
  sorry

end actual_speed_proof_l1185_118579


namespace product_of_fraction_parts_l1185_118548

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.018018018018018018018018018018018018018018018018018

/-- Express the repeating decimal as a fraction in lowest terms -/
def decimal_to_fraction (d : ℚ) : ℚ := d

/-- Calculate the product of numerator and denominator of a fraction -/
def numerator_denominator_product (q : ℚ) : ℕ :=
  (q.num.natAbs) * (q.den)

/-- Theorem stating that the product of numerator and denominator of 0.018̅ in lowest terms is 222 -/
theorem product_of_fraction_parts : 
  numerator_denominator_product (decimal_to_fraction repeating_decimal) = 222 := by
  sorry

end product_of_fraction_parts_l1185_118548


namespace farmer_cows_problem_l1185_118516

theorem farmer_cows_problem (initial_cows : ℕ) : 
  (3 / 4 : ℚ) * (initial_cows + 5 : ℚ) = 42 → initial_cows = 51 :=
by
  sorry

end farmer_cows_problem_l1185_118516


namespace inequality_solution_set_inequality_positive_reals_l1185_118509

-- Part 1: Inequality solution set
theorem inequality_solution_set (x : ℝ) :
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1/2 < x ∧ x ≤ 1 :=
sorry

-- Part 2: Inequality with positive real numbers
theorem inequality_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/(b + c)) ≥ 4 :=
sorry

end inequality_solution_set_inequality_positive_reals_l1185_118509


namespace krishans_money_krishan_has_4046_l1185_118540

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount, 
    calculate Krishan's amount. -/
theorem krishans_money 
  (ram_gopal_ratio : ℚ) 
  (gopal_krishan_ratio : ℚ) 
  (ram_money : ℕ) : ℕ :=
  let gopal_money := (ram_money * 17) / 7
  let krishan_money := (gopal_money * 17) / 7
  krishan_money

/-- Prove that Krishan has Rs. 4046 given the problem conditions. -/
theorem krishan_has_4046 :
  krishans_money (7/17) (7/17) 686 = 4046 := by
  sorry

end krishans_money_krishan_has_4046_l1185_118540


namespace division_problem_l1185_118577

theorem division_problem :
  ∃ (quotient : ℕ), 136 = 15 * quotient + 1 ∧ quotient = 9 := by
  sorry

end division_problem_l1185_118577


namespace tan_alpha_and_expression_l1185_118593

theorem tan_alpha_and_expression (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 ∧ (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7/6 := by
  sorry

end tan_alpha_and_expression_l1185_118593


namespace floor_abs_negative_l1185_118512

theorem floor_abs_negative : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry

end floor_abs_negative_l1185_118512


namespace library_books_count_l1185_118591

/-- The number of books in a library after two years of purchases -/
def library_books (initial_books : ℕ) (books_last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial_books + books_last_year + multiplier * books_last_year

/-- Theorem stating that the library now has 300 books -/
theorem library_books_count : library_books 100 50 3 = 300 := by
  sorry

end library_books_count_l1185_118591


namespace fraction_equals_zero_l1185_118505

theorem fraction_equals_zero (x : ℝ) : 
  (x^2 - 1) / (1 - x) = 0 ∧ 1 - x ≠ 0 → x = -1 := by
  sorry

end fraction_equals_zero_l1185_118505


namespace jacksons_decorations_l1185_118529

/-- Given that Mrs. Jackson has 4 boxes of Christmas decorations with 15 decorations in each box
    and she used 35 decorations, prove that she gave 25 decorations to her neighbor. -/
theorem jacksons_decorations (num_boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ)
    (h1 : num_boxes = 4)
    (h2 : decorations_per_box = 15)
    (h3 : used_decorations = 35) :
    num_boxes * decorations_per_box - used_decorations = 25 := by
  sorry

end jacksons_decorations_l1185_118529


namespace smallest_sqrt_x_minus_one_l1185_118513

theorem smallest_sqrt_x_minus_one :
  ∀ x : ℝ, 
    (Real.sqrt (x - 1) ≥ 0) ∧ 
    (Real.sqrt (x - 1) = 0 ↔ x = 1) :=
by sorry

end smallest_sqrt_x_minus_one_l1185_118513


namespace function_with_period_3_is_periodic_l1185_118575

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the periodicity condition
def is_periodic_with_period (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- State the theorem
theorem function_with_period_3_is_periodic :
  (∀ x, f (x + 3) = f x) → ∃ p > 0, is_periodic_with_period f p :=
sorry

end function_with_period_3_is_periodic_l1185_118575


namespace factorization_proof_l1185_118566

theorem factorization_proof (m : ℝ) : 4 - m^2 = (2 + m) * (2 - m) := by
  sorry

end factorization_proof_l1185_118566


namespace resulting_polygon_has_16_sides_l1185_118514

/-- Represents a regular polygon --/
structure RegularPolygon where
  sides : ℕ
  sides_positive : sides > 0

/-- The resulting polygon formed by connecting the given regular polygons --/
def resulting_polygon (triangle square pentagon heptagon hexagon octagon : RegularPolygon) : ℕ :=
  2 + 2 + (4 * 3)

/-- Theorem stating that the resulting polygon has 16 sides --/
theorem resulting_polygon_has_16_sides 
  (triangle : RegularPolygon) 
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (hexagon : RegularPolygon)
  (octagon : RegularPolygon)
  (h1 : triangle.sides = 3)
  (h2 : square.sides = 4)
  (h3 : pentagon.sides = 5)
  (h4 : heptagon.sides = 7)
  (h5 : hexagon.sides = 6)
  (h6 : octagon.sides = 8) :
  resulting_polygon triangle square pentagon heptagon hexagon octagon = 16 := by
  sorry

end resulting_polygon_has_16_sides_l1185_118514


namespace lawn_mowing_l1185_118551

theorem lawn_mowing (mary_rate tom_rate : ℚ)
  (h1 : mary_rate = 1 / 3)
  (h2 : tom_rate = 1 / 6)
  (total_lawn : ℚ)
  (h3 : total_lawn = 1) :
  let combined_rate := mary_rate + tom_rate
  let mowed_together := combined_rate * 1
  let mowed_mary_alone := mary_rate * 1
  let total_mowed := mowed_together + mowed_mary_alone
  total_lawn - total_mowed = 1 / 6 := by
sorry

end lawn_mowing_l1185_118551


namespace expression_simplification_l1185_118584

theorem expression_simplification (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 8) : 
  y * (5 * x - 4 * y) + (x - 2 * y)^2 = 17/4 := by
  sorry

end expression_simplification_l1185_118584


namespace johns_water_usage_l1185_118585

/-- Calculates the total water usage for John's showers over 4 weeks -/
def total_water_usage (weeks : ℕ) (shower_frequency : ℕ) (shower_duration : ℕ) (water_per_minute : ℕ) : ℕ :=
  let days := weeks * 7
  let num_showers := days / shower_frequency
  let water_per_shower := shower_duration * water_per_minute
  num_showers * water_per_shower

/-- Proves that John's total water usage over 4 weeks is 280 gallons -/
theorem johns_water_usage : total_water_usage 4 2 10 2 = 280 := by
  sorry

end johns_water_usage_l1185_118585


namespace max_ab_value_l1185_118543

theorem max_ab_value (a b c : ℝ) : 
  (∀ x : ℝ, 2*x + 2 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 2*x^2 - 2*x + 4) →
  a*b ≤ (1/2 : ℝ) :=
by sorry

end max_ab_value_l1185_118543


namespace coin_difference_l1185_118562

def coin_values : List Nat := [1, 5, 10, 25, 50]
def target_amount : Nat := 65

def min_coins (amount : Nat) (coins : List Nat) : Nat :=
  sorry

def max_coins (amount : Nat) (coins : List Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins target_amount coin_values - min_coins target_amount coin_values = 62 :=
by sorry

end coin_difference_l1185_118562


namespace sum_of_three_numbers_l1185_118598

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a*b + b*c + a*c = 100) : 
  a + b + c = 21 := by
sorry

end sum_of_three_numbers_l1185_118598


namespace millipede_segment_ratio_l1185_118526

/-- Proves that the ratio of segments of two unknown-length millipedes to a 60-segment millipede is 4:1 --/
theorem millipede_segment_ratio : 
  ∀ (x : ℕ), -- x represents the number of segments in each of the two unknown-length millipedes
  (2 * x + 60 + 500 = 800) → -- Total segments equation
  ((2 * x) : ℚ) / 60 = 4 / 1 := by
sorry

end millipede_segment_ratio_l1185_118526


namespace rhombus_longer_diagonal_l1185_118549

/-- A rhombus with side length 65 and shorter diagonal 72 has a longer diagonal of 108 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 65 → shorter_diag = 72 → longer_diag = 108 → 
  side^2 = (shorter_diag / 2)^2 + (longer_diag / 2)^2 := by
sorry

end rhombus_longer_diagonal_l1185_118549


namespace walking_rate_ratio_l1185_118592

theorem walking_rate_ratio (usual_time new_time distance : ℝ) 
  (h1 : usual_time = 36)
  (h2 : new_time = usual_time - 4)
  (h3 : distance > 0)
  (h4 : usual_time > 0)
  (h5 : new_time > 0) :
  (distance / new_time) / (distance / usual_time) = 9 / 8 := by
sorry

end walking_rate_ratio_l1185_118592


namespace deepak_age_l1185_118533

theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) (years_ahead : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 2 →
  rahul_future_age = 26 →
  years_ahead = 10 →
  ∃ (x : ℕ), rahul_ratio * x + years_ahead = rahul_future_age ∧ deepak_ratio * x = 8 :=
by sorry

end deepak_age_l1185_118533


namespace expand_expression_l1185_118550

theorem expand_expression (x y : ℝ) : (3*x - 15) * (4*y + 20) = 12*x*y + 60*x - 60*y - 300 := by
  sorry

end expand_expression_l1185_118550


namespace bamboo_pole_is_ten_feet_l1185_118515

/-- The length of a bamboo pole satisfying specific conditions relative to a door --/
def bamboo_pole_length : ℝ → Prop := fun x =>
  ∃ (door_width door_height : ℝ),
    door_width > 0 ∧ 
    door_height > 0 ∧ 
    x = door_width + 4 ∧ 
    x = door_height + 2 ∧ 
    x^2 = door_width^2 + door_height^2

/-- Theorem stating that the bamboo pole length is 10 feet --/
theorem bamboo_pole_is_ten_feet : 
  bamboo_pole_length 10 := by
  sorry

#check bamboo_pole_is_ten_feet

end bamboo_pole_is_ten_feet_l1185_118515


namespace f_g_deriv_signs_l1185_118557

-- Define f and g as real-valued functions
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → deriv g x > 0

-- State the theorem
theorem f_g_deriv_signs :
  ∀ x : ℝ, x < 0 → deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end f_g_deriv_signs_l1185_118557


namespace sum_of_3rd_4th_5th_terms_l1185_118589

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem sum_of_3rd_4th_5th_terms
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : geometric_sequence a)
  (h_ratio : ∃ (r : ℝ), ∀ n, a (n + 1) = 2 * a n)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end sum_of_3rd_4th_5th_terms_l1185_118589


namespace factorization_mx_minus_my_l1185_118531

theorem factorization_mx_minus_my (m x y : ℝ) : m * x - m * y = m * (x - y) := by
  sorry

end factorization_mx_minus_my_l1185_118531


namespace complex_fourth_power_l1185_118582

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fourth_power : (1 - i) ^ 4 = -4 := by
  sorry

end complex_fourth_power_l1185_118582


namespace parabola_shift_theorem_l1185_118555

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (right : ℝ) (down : ℝ) : Parabola :=
  { a := p.a
    h := p.h + right
    k := p.k - down }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 4 ∧ p.k = 2 →
  (shift p 1 3).a = 3 ∧ (shift p 1 3).h = 5 ∧ (shift p 1 3).k = -1 := by
  sorry

end parabola_shift_theorem_l1185_118555


namespace square_of_negative_square_l1185_118594

theorem square_of_negative_square (m : ℝ) : (-m^2)^2 = m^4 := by
  sorry

end square_of_negative_square_l1185_118594


namespace perpendicular_tangents_circles_l1185_118506

/-- Two circles with perpendicular tangents at intersection points -/
theorem perpendicular_tangents_circles (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*y = 0 ∧ x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0 →
    ∃ m n : ℝ, 
      m^2 + n^2 + 4*n = 0 ∧
      2*(a-1)*m - 2*n + a^2 = 0 ∧
      (n + 2) / m * (n + 1) / (m - (1 - a)) = -1) →
  a = -2 :=
sorry

end perpendicular_tangents_circles_l1185_118506


namespace prism_diagonal_angle_l1185_118574

/-- Given a right prism with a right triangular base, where one acute angle of the base is α
    and the largest lateral face is a square, this theorem states that the angle β between
    the intersecting diagonals of the other two lateral faces is arccos(2 / √(8 + sin²(2α))) -/
theorem prism_diagonal_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) :
  ∃ (β : ℝ),
    β = Real.arccos (2 / Real.sqrt (8 + Real.sin (2 * α) ^ 2)) ∧
    0 ≤ β ∧
    β ≤ π :=
sorry

end prism_diagonal_angle_l1185_118574


namespace sum_digits_first_2002_even_integers_l1185_118560

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The nth positive even integer -/
def nthEvenInteger (n : ℕ) : ℕ := sorry

/-- The sum of digits for the first n positive even integers -/
def sumDigitsFirstNEvenIntegers (n : ℕ) : ℕ := sorry

/-- Theorem: The total number of digits used to write the first 2002 positive even integers is 7456 -/
theorem sum_digits_first_2002_even_integers : 
  sumDigitsFirstNEvenIntegers 2002 = 7456 := by sorry

end sum_digits_first_2002_even_integers_l1185_118560


namespace length_PQ_value_l1185_118583

/-- Triangle ABC with given side lengths and angle bisectors --/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- AH is altitude
  AH : ℝ
  -- Q and P are intersection points of angle bisectors with altitude
  AQ : ℝ
  AP : ℝ
  -- Conditions
  side_lengths : AB = 6 ∧ BC = 10 ∧ CA = 8
  altitude : AH = 4.8
  angle_bisector_intersections : AQ = 20/3 ∧ AP = 3

/-- The length of PQ in the given triangle configuration --/
def length_PQ (t : TriangleABC) : ℝ := t.AQ - t.AP

/-- Theorem stating that the length of PQ is 3.67 --/
theorem length_PQ_value (t : TriangleABC) : length_PQ t = 3.67 := by
  sorry

end length_PQ_value_l1185_118583


namespace arithmetic_mean_after_removal_l1185_118510

theorem arithmetic_mean_after_removal (s : Finset ℝ) (a b c : ℝ) :
  s.card = 80 →
  a = 50 ∧ b = 60 ∧ c = 70 →
  a ∈ s ∧ b ∈ s ∧ c ∈ s →
  (s.sum id) / s.card = 45 →
  ((s.sum id) - (a + b + c)) / (s.card - 3) = 3420 / 77 :=
by sorry

end arithmetic_mean_after_removal_l1185_118510


namespace fraction_equality_l1185_118508

theorem fraction_equality (a b : ℕ) (h1 : a + b = 1210) (h2 : b = 484) :
  (4 / 15 : ℚ) * a = (2 / 5 : ℚ) * b :=
sorry

end fraction_equality_l1185_118508


namespace symmetry_of_sine_function_l1185_118588

/-- Given a function f(x) = sin(wx + π/4) where w > 0 and 
    the minimum positive period of f(x) is π, 
    prove that the graph of f(x) is symmetrical about the line x = π/8 -/
theorem symmetry_of_sine_function (w : ℝ) (h1 : w > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (w * x + π / 4)
  (∀ x : ℝ, f (x + π) = f x) →  -- minimum positive period is π
  ∀ x : ℝ, f (π / 4 - x) = f (π / 4 + x) := by
sorry

end symmetry_of_sine_function_l1185_118588


namespace set_equality_l1185_118502

-- Define sets A and B
def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Define the set we want to prove equal to our result
def S : Set ℝ := {x | x ∈ A ∧ x ∉ A ∩ B}

-- State the theorem
theorem set_equality : S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end set_equality_l1185_118502


namespace water_added_proof_l1185_118569

def container_problem (capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : Prop :=
  let initial_volume := capacity * initial_fill
  let final_volume := capacity * final_fill
  final_volume - initial_volume = 20

theorem water_added_proof :
  container_problem 80 0.5 0.75 :=
sorry

end water_added_proof_l1185_118569


namespace payment_calculation_l1185_118503

/-- The payment for C given the work rates of A and B, total payment, and total work days -/
def payment_for_C (a_rate : ℚ) (b_rate : ℚ) (total_payment : ℚ) (total_days : ℚ) : ℚ :=
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * total_days
  let c_work := 1 - ab_work
  c_work * total_payment

theorem payment_calculation (a_rate b_rate total_payment total_days : ℚ) 
  (ha : a_rate = 1/6)
  (hb : b_rate = 1/8)
  (hp : total_payment = 3360)
  (hd : total_days = 3) :
  payment_for_C a_rate b_rate total_payment total_days = 420 := by
  sorry

#eval payment_for_C (1/6) (1/8) 3360 3

end payment_calculation_l1185_118503


namespace cubic_equation_transformation_l1185_118525

theorem cubic_equation_transformation (A B C : ℝ) :
  ∃ (p q β : ℝ), ∀ (z x : ℝ),
    (z^3 + A * z^2 + B * z + C = 0) ↔
    (z = x + β ∧ x^3 + p * x + q = 0) :=
by sorry

end cubic_equation_transformation_l1185_118525


namespace tangerine_persimmon_ratio_l1185_118539

theorem tangerine_persimmon_ratio :
  let apples : ℕ := 24
  let tangerines : ℕ := 6 * apples
  let persimmons : ℕ := 8
  tangerines = 18 * persimmons :=
by
  sorry

end tangerine_persimmon_ratio_l1185_118539


namespace oil_mixture_price_l1185_118528

/-- Given two types of oil mixed together, calculate the price of the second oil. -/
theorem oil_mixture_price (volume1 volume2 total_volume : ℚ) (price1 mixture_price : ℚ) :
  volume1 = 10 →
  volume2 = 5 →
  total_volume = volume1 + volume2 →
  price1 = 54 →
  mixture_price = 58 →
  ∃ price2 : ℚ, 
    price2 = 66 ∧
    volume1 * price1 + volume2 * price2 = total_volume * mixture_price :=
by sorry

end oil_mixture_price_l1185_118528


namespace a_eq_b_sufficient_a_eq_b_not_necessary_l1185_118542

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - x + a ≤ 0}
def B (b : ℝ) : Set ℝ := {x : ℝ | x^2 - x + b ≤ 0}

-- Theorem stating that a = b is sufficient for A = B
theorem a_eq_b_sufficient (a b : ℝ) : a = b → A a = B b := by sorry

-- Theorem stating that a = b is not necessary for A = B
theorem a_eq_b_not_necessary : ∃ a b : ℝ, A a = B b ∧ a ≠ b := by sorry

end a_eq_b_sufficient_a_eq_b_not_necessary_l1185_118542


namespace max_value_abc_l1185_118547

theorem max_value_abc (a b c : ℕ+) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a * b * c = 16) :
  (∀ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 16 →
    x ^ y.val - y ^ z.val + z ^ x.val ≤ a ^ b.val - b ^ c.val + c ^ a.val) →
  a ^ b.val - b ^ c.val + c ^ a.val = 263 :=
by sorry

end max_value_abc_l1185_118547


namespace marble_probability_l1185_118568

def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 6
def green_choices : ℕ := 3

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

theorem marble_probability : 
  (Nat.choose trials green_choices : ℚ) * 
  (prob_green ^ green_choices) * 
  (prob_purple ^ (trials - green_choices)) * 
  prob_purple = 4913248/34171875 := by sorry

end marble_probability_l1185_118568


namespace carpet_shaded_area_l1185_118595

theorem carpet_shaded_area (S T : ℝ) : 
  12 / S = 4 →
  S / T = 2 →
  S > 0 →
  T > 0 →
  S^2 + 8 * T^2 = 27 := by
sorry

end carpet_shaded_area_l1185_118595


namespace debate_tournament_participants_l1185_118545

theorem debate_tournament_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 30 → initial_participants = 300 := by
  sorry

end debate_tournament_participants_l1185_118545


namespace no_intersection_l1185_118546

/-- Definition of a parabola -/
def is_parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Definition of a point inside the parabola -/
def is_inside_parabola (x₀ y₀ : ℝ) : Prop := y₀^2 < 4*x₀

/-- Definition of the line -/
def line_equation (x₀ y₀ x y : ℝ) : Prop := y₀*y = 2*(x + x₀)

/-- Theorem stating that a line passing through a point inside the parabola has no intersection with the parabola -/
theorem no_intersection (x₀ y₀ : ℝ) :
  is_inside_parabola x₀ y₀ →
  ∀ x y : ℝ, is_parabola x y ∧ line_equation x₀ y₀ x y → False :=
sorry

end no_intersection_l1185_118546


namespace balls_in_original_positions_l1185_118586

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The number of transpositions performed -/
def num_transpositions : ℕ := 3

/-- The probability of a ball being in its original position after the transpositions -/
def prob_original_position : ℚ := 127 / 343

/-- The expected number of balls in their original positions after the transpositions -/
def expected_original_positions : ℚ := 889 / 343

theorem balls_in_original_positions :
  num_balls * prob_original_position = expected_original_positions := by sorry

end balls_in_original_positions_l1185_118586


namespace half_minus_quarter_equals_two_l1185_118537

theorem half_minus_quarter_equals_two (n : ℝ) : n = 8 → (0.5 * n) - (0.25 * n) = 2 := by
  sorry

end half_minus_quarter_equals_two_l1185_118537


namespace monotonic_increase_intervals_l1185_118571

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem monotonic_increase_intervals (x : ℝ) :
  StrictMonoOn f (Set.Iio (-1)) ∧ StrictMonoOn f (Set.Ioi 1) :=
sorry

end monotonic_increase_intervals_l1185_118571


namespace i_power_difference_zero_l1185_118570

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_power_difference_zero : i^45 - i^305 = 0 := by
  sorry

end i_power_difference_zero_l1185_118570


namespace notebook_distribution_l1185_118572

/-- Proves that the ratio of notebooks per child to the number of children is 1:8 
    given the conditions in the problem. -/
theorem notebook_distribution (C : ℕ) (N : ℚ) : 
  (∃ (k : ℕ), N = k / C) →  -- Number of notebooks each child got is a fraction of number of children
  (16 = 2 * k / C) →        -- If number of children halved, each would get 16 notebooks
  (C * N = 512) →           -- Total notebooks distributed is 512
  N / C = 1 / 8 :=          -- Ratio of notebooks per child to number of children is 1:8
by sorry

end notebook_distribution_l1185_118572


namespace tank_capacity_l1185_118596

theorem tank_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (total_capacity : ℕ) : 
  num_trucks = 3 → 
  tanks_per_truck = 3 → 
  total_capacity = 1350 → 
  (total_capacity / (num_trucks * tanks_per_truck) : ℚ) = 150 := by
sorry

end tank_capacity_l1185_118596


namespace stream_speed_l1185_118522

/-- Given a river with stream speed v and a rower with speed u in still water,
    if the rower travels 27 km upstream and 81 km downstream, each in 9 hours,
    then the speed of the stream v is 3 km/h. -/
theorem stream_speed (v u : ℝ) 
  (h1 : 27 / (u - v) = 9)  -- Upstream condition
  (h2 : 81 / (u + v) = 9)  -- Downstream condition
  : v = 3 := by
  sorry


end stream_speed_l1185_118522


namespace yz_minus_zx_minus_xy_l1185_118527

theorem yz_minus_zx_minus_xy (x y z : ℝ) 
  (h1 : x - y - z = 19) 
  (h2 : x^2 + y^2 + z^2 ≠ 19) : 
  y*z - z*x - x*y = 171 := by sorry

end yz_minus_zx_minus_xy_l1185_118527


namespace smallest_number_with_remainders_l1185_118517

theorem smallest_number_with_remainders : ∃ (a : ℕ), 
  (a % 3 = 2) ∧ (a % 5 = 3) ∧ (a % 7 = 3) ∧
  (∀ (b : ℕ), b < a → ¬((b % 3 = 2) ∧ (b % 5 = 3) ∧ (b % 7 = 3))) ∧
  a = 98 := by
  sorry

end smallest_number_with_remainders_l1185_118517


namespace min_value_x_plus_y_l1185_118538

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 2) :
  ∃ (m : ℝ), (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 2 → x + y ≤ a + b) ∧ x + y = m ∧ m = 8 :=
sorry

end min_value_x_plus_y_l1185_118538


namespace inserted_numbers_sum_l1185_118559

theorem inserted_numbers_sum (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ 
  2 < a ∧ a < b ∧ b < 12 ∧
  (∃ r : ℝ, r > 0 ∧ a = 2 * r ∧ b = 2 * r^2) ∧
  (∃ d : ℝ, b = a + d ∧ 12 = b + d) →
  a + b = 12 := by
sorry

end inserted_numbers_sum_l1185_118559


namespace equation1_unique_solution_equation2_no_solution_l1185_118563

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  5 / (2 * x) - 1 / (x - 3) = 0

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  1 / (x - 2) = 4 / (x^2 - 4)

-- Theorem for the first equation
theorem equation1_unique_solution :
  ∃! x : ℝ, equation1 x ∧ x = 5 :=
sorry

-- Theorem for the second equation
theorem equation2_no_solution :
  ∀ x : ℝ, ¬ equation2 x :=
sorry

end equation1_unique_solution_equation2_no_solution_l1185_118563


namespace point_on_600_degree_angle_l1185_118521

/-- If a point (-4, a) lies on the terminal side of an angle of 600°, then a = -4√3 -/
theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * π / 180 ∧ Real.tan θ = a / (-4)) → a = -4 * Real.sqrt 3 := by
  sorry

end point_on_600_degree_angle_l1185_118521


namespace distance_XY_is_80_l1185_118552

/-- The distance from X to Y in miles. -/
def distance_XY : ℝ := 80

/-- Yolanda's walking rate in miles per hour. -/
def yolanda_rate : ℝ := 8

/-- Bob's walking rate in miles per hour. -/
def bob_rate : ℝ := 9

/-- The distance Bob walked when they met, in miles. -/
def bob_distance : ℝ := 38.11764705882353

/-- The time difference between Yolanda and Bob's start times, in hours. -/
def time_difference : ℝ := 1

theorem distance_XY_is_80 :
  distance_XY = yolanda_rate * (time_difference + bob_distance / bob_rate) + bob_distance :=
sorry

end distance_XY_is_80_l1185_118552


namespace sum_of_cubes_l1185_118520

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 := by
  sorry

end sum_of_cubes_l1185_118520


namespace total_samplers_percentage_l1185_118518

/-- Represents the percentage of customers for a specific candy type -/
structure CandyData where
  caught : ℝ
  notCaught : ℝ

/-- Represents the data for all candy types -/
structure CandyStore where
  A : CandyData
  B : CandyData
  C : CandyData
  D : CandyData

/-- Calculates the total percentage of customers who sample any type of candy -/
def totalSamplers (store : CandyStore) : ℝ :=
  store.A.caught + store.A.notCaught +
  store.B.caught + store.B.notCaught +
  store.C.caught + store.C.notCaught +
  store.D.caught + store.D.notCaught

/-- The candy store data -/
def candyStoreData : CandyStore :=
  { A := { caught := 12, notCaught := 7 }
    B := { caught := 5,  notCaught := 6 }
    C := { caught := 9,  notCaught := 3 }
    D := { caught := 4,  notCaught := 8 } }

theorem total_samplers_percentage :
  totalSamplers candyStoreData = 54 := by sorry

end total_samplers_percentage_l1185_118518


namespace line_translation_l1185_118534

/-- Represents a line in the 2D Cartesian plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The vertical translation distance between two lines -/
def vertical_translation (l1 l2 : Line) : ℝ :=
  l2.y_intercept - l1.y_intercept

theorem line_translation (l1 l2 : Line) :
  l1.slope = -2 ∧ l1.y_intercept = -2 ∧ 
  l2.slope = -2 ∧ l2.y_intercept = 4 →
  vertical_translation l1 l2 = 6 := by
  sorry

end line_translation_l1185_118534


namespace towel_shrinkage_l1185_118578

theorem towel_shrinkage (L B : ℝ) (h_positive : L > 0 ∧ B > 0) : 
  let new_length := 0.8 * L
  let new_area := 0.72 * (L * B)
  ∃ (new_breadth : ℝ), new_area = new_length * new_breadth ∧ new_breadth = 0.9 * B :=
sorry

end towel_shrinkage_l1185_118578


namespace supplement_not_always_greater_l1185_118500

/-- The supplement of an angle (in degrees) -/
def supplement (x : ℝ) : ℝ := 180 - x

/-- Theorem stating that the statement "The supplement of an angle is always greater than the angle itself" is false -/
theorem supplement_not_always_greater (x : ℝ) : ¬ (∀ x, supplement x > x) := by
  sorry

end supplement_not_always_greater_l1185_118500


namespace batsman_running_fraction_l1185_118523

/-- Represents the score of a batsman in cricket --/
structure BatsmanScore where
  total_runs : ℕ
  boundaries : ℕ
  sixes : ℕ

/-- Calculates the fraction of runs made by running between wickets --/
def runningFraction (score : BatsmanScore) : ℚ :=
  let boundary_runs := 4 * score.boundaries
  let six_runs := 6 * score.sixes
  let running_runs := score.total_runs - (boundary_runs + six_runs)
  (running_runs : ℚ) / score.total_runs

theorem batsman_running_fraction :
  let score : BatsmanScore := ⟨250, 15, 10⟩
  runningFraction score = 13 / 25 := by
  sorry

end batsman_running_fraction_l1185_118523


namespace x_coordinate_range_l1185_118511

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 6

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_M x y

-- Define a point on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem x_coordinate_range :
  ∀ (A B C : ℝ × ℝ),
  point_on_line A.1 A.2 →
  point_on_circle B.1 B.2 →
  point_on_circle C.1 C.2 →
  angle A B C = π/3 →
  1 ≤ A.1 ∧ A.1 ≤ 5 :=
by sorry

end x_coordinate_range_l1185_118511


namespace greatest_base_nine_digit_sum_l1185_118541

def base_nine_digit_sum (n : ℕ) : ℕ :=
  (n.digits 9).sum

theorem greatest_base_nine_digit_sum :
  ∃ (m : ℕ), m < 2500 ∧ base_nine_digit_sum m = 24 ∧
  ∀ (n : ℕ), n < 2500 → base_nine_digit_sum n ≤ 24 := by
  sorry

end greatest_base_nine_digit_sum_l1185_118541


namespace oranges_picked_total_l1185_118567

/-- The number of oranges Joan picked -/
def joan_oranges : ℕ := 37

/-- The number of oranges Sara picked -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := joan_oranges + sara_oranges

theorem oranges_picked_total :
  total_oranges = 47 := by sorry

end oranges_picked_total_l1185_118567


namespace trapezoid_area_l1185_118519

/-- The area of a trapezoid with given base lengths and leg lengths -/
theorem trapezoid_area (b1 b2 l1 l2 : ℝ) (h : ℝ) 
  (hb1 : b1 = 10) 
  (hb2 : b2 = 21) 
  (hl1 : l1 = Real.sqrt 34) 
  (hl2 : l2 = 3 * Real.sqrt 5) 
  (hh : h^2 + 5^2 = 34) : 
  (b1 + b2) * h / 2 = 93 / 2 := by
  sorry

#check trapezoid_area

end trapezoid_area_l1185_118519
