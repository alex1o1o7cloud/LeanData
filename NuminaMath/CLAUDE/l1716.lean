import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1716_171603

theorem unique_solution_for_equation (y : ℝ) : y + 49 / y = 14 ↔ y = 7 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1716_171603


namespace NUMINAMATH_CALUDE_equation_solutions_l1716_171622

def equation (x : ℝ) : Prop :=
  6 / (Real.sqrt (x - 8) - 9) + 1 / (Real.sqrt (x - 8) - 4) + 
  7 / (Real.sqrt (x - 8) + 4) + 12 / (Real.sqrt (x - 8) + 9) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {17, 44} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1716_171622


namespace NUMINAMATH_CALUDE_problem_solution_l1716_171673

theorem problem_solution :
  (∀ x : ℝ, x^2 + x + 2 ≥ 0) ∧
  (∀ x y : ℝ, x * y = ((x + y) / 2)^2 ↔ x = y) ∧
  (∃ p q : Prop, ¬(p ∧ q) ∧ ¬(¬p ∧ ¬q)) ∧
  (∀ A B C : ℝ, ∀ sinA sinB : ℝ, 
    sinA = Real.sin A ∧ sinB = Real.sin B →
    sinA > sinB → A > B) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1716_171673


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1716_171648

/-- Given an ellipse C and a line l, prove that under certain conditions, 
    a point derived from l lies on a specific circle. -/
theorem ellipse_line_intersection (a b : ℝ) (k m : ℝ) : 
  a > b ∧ b > 0 ∧ 
  (a^2 - b^2) / a^2 = 3 / 4 ∧
  b = 1 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧
    x₂^2 / a^2 + y₂^2 / b^2 = 1 ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    (y₁ * x₂) / (x₁ * y₂) = 5 / 4) →
  m^2 + k^2 = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1716_171648


namespace NUMINAMATH_CALUDE_count_arrangements_l1716_171684

/-- Represents the number of students --/
def totalStudents : ℕ := 6

/-- Represents the number of boys --/
def numBoys : ℕ := 3

/-- Represents the number of girls --/
def numGirls : ℕ := 3

/-- Represents whether girls are allowed at the ends --/
def girlsAtEnds : Prop := False

/-- Represents whether girls A and B can stand next to girl C --/
def girlsABNextToC : Prop := False

/-- The number of valid arrangements --/
def validArrangements : ℕ := 72

/-- Theorem stating the number of valid arrangements --/
theorem count_arrangements :
  (totalStudents = numBoys + numGirls) →
  (numBoys = 3) →
  (numGirls = 3) →
  girlsAtEnds = False →
  girlsABNextToC = False →
  validArrangements = 72 := by
  sorry

end NUMINAMATH_CALUDE_count_arrangements_l1716_171684


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1716_171611

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = 1 / 3) : 
  Real.cos α = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1716_171611


namespace NUMINAMATH_CALUDE_solution_range_l1716_171638

theorem solution_range (m : ℝ) : 
  (∃ x y : ℝ, x + y = -1 ∧ 5 * x + 2 * y = 6 * m + 7 ∧ 2 * x - y < 19) → 
  m < 3/2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l1716_171638


namespace NUMINAMATH_CALUDE_cab_delay_l1716_171662

/-- Proves that a cab with reduced speed arrives 15 minutes late -/
theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) : 
  usual_time = 75 → speed_ratio = 5/6 → 
  (usual_time / speed_ratio) - usual_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_cab_delay_l1716_171662


namespace NUMINAMATH_CALUDE_chords_from_eight_points_l1716_171690

/-- The number of chords that can be drawn from n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 8 points on a circle's circumference is 28 -/
theorem chords_from_eight_points : num_chords 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_chords_from_eight_points_l1716_171690


namespace NUMINAMATH_CALUDE_carpet_width_l1716_171659

/-- Proves that a rectangular carpet covering 30% of a 120 square feet floor with a length of 9 feet has a width of 4 feet. -/
theorem carpet_width (floor_area : ℝ) (carpet_coverage : ℝ) (carpet_length : ℝ) :
  floor_area = 120 →
  carpet_coverage = 0.3 →
  carpet_length = 9 →
  (floor_area * carpet_coverage) / carpet_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l1716_171659


namespace NUMINAMATH_CALUDE_prob_one_of_each_specific_jar_l1716_171616

/-- Represents the number of marbles of each color in the jar -/
structure MarbleJar :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)

/-- Calculates the probability of drawing one red, one blue, and one yellow marble -/
def prob_one_of_each (jar : MarbleJar) : ℚ :=
  sorry

/-- The theorem statement -/
theorem prob_one_of_each_specific_jar :
  prob_one_of_each ⟨3, 8, 9⟩ = 18 / 95 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_of_each_specific_jar_l1716_171616


namespace NUMINAMATH_CALUDE_x_fourth_equals_one_l1716_171687

theorem x_fourth_equals_one (x : ℝ) 
  (h : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = Real.sqrt 2) : 
  x^4 = 1 := by
sorry

end NUMINAMATH_CALUDE_x_fourth_equals_one_l1716_171687


namespace NUMINAMATH_CALUDE_abc_inequality_l1716_171656

theorem abc_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1716_171656


namespace NUMINAMATH_CALUDE_sarah_shopping_theorem_l1716_171619

theorem sarah_shopping_theorem (toy_car1 toy_car2_orig scarf_orig beanie gloves book necklace_orig : ℚ)
  (toy_car2_discount scarf_discount beanie_tax necklace_discount : ℚ)
  (remaining : ℚ)
  (h1 : toy_car1 = 12)
  (h2 : toy_car2_orig = 15)
  (h3 : toy_car2_discount = 0.1)
  (h4 : scarf_orig = 10)
  (h5 : scarf_discount = 0.2)
  (h6 : beanie = 14)
  (h7 : beanie_tax = 0.08)
  (h8 : necklace_orig = 20)
  (h9 : necklace_discount = 0.05)
  (h10 : gloves = 12)
  (h11 : book = 15)
  (h12 : remaining = 7) :
  toy_car1 +
  (toy_car2_orig - toy_car2_orig * toy_car2_discount) +
  (scarf_orig - scarf_orig * scarf_discount) +
  (beanie + beanie * beanie_tax) +
  (necklace_orig - necklace_orig * necklace_discount) +
  gloves +
  book +
  remaining = 101.62 := by
sorry

end NUMINAMATH_CALUDE_sarah_shopping_theorem_l1716_171619


namespace NUMINAMATH_CALUDE_rented_cars_at_3600_optimal_rent_max_monthly_revenue_l1716_171699

/-- Represents the rental company's car fleet and pricing model. -/
structure RentalCompany where
  totalCars : ℕ := 100
  initialRent : ℕ := 3000
  rentIncrease : ℕ := 50
  maintenanceCostRented : ℕ := 150
  maintenanceCostUnrented : ℕ := 50

/-- Calculates the number of rented cars given a specific rent. -/
def rentedCars (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.initialRent) / company.rentIncrease

/-- Calculates the monthly revenue for the rental company. -/
def monthlyRevenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := rentedCars company rent
  rented * (rent - company.maintenanceCostRented) -
    (company.totalCars - rented) * company.maintenanceCostUnrented

/-- Theorem stating the correct number of rented cars at 3600 yuan rent. -/
theorem rented_cars_at_3600 (company : RentalCompany) :
    rentedCars company 3600 = 88 := by sorry

/-- Theorem stating the optimal rent that maximizes revenue. -/
theorem optimal_rent (company : RentalCompany) :
    ∃ (optimalRent : ℕ), optimalRent = 4050 ∧
    ∀ (rent : ℕ), monthlyRevenue company rent ≤ monthlyRevenue company optimalRent := by sorry

/-- Theorem stating the maximum monthly revenue. -/
theorem max_monthly_revenue (company : RentalCompany) :
    ∃ (maxRevenue : ℕ), maxRevenue = 307050 ∧
    ∀ (rent : ℕ), monthlyRevenue company rent ≤ maxRevenue := by sorry

end NUMINAMATH_CALUDE_rented_cars_at_3600_optimal_rent_max_monthly_revenue_l1716_171699


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1716_171640

theorem polar_to_rectangular_conversion :
  ∀ (x y ρ θ : ℝ),
  ρ = Real.sin θ + Real.cos θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  ρ^2 = x^2 + y^2 →
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1716_171640


namespace NUMINAMATH_CALUDE_f_four_times_one_l1716_171626

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_four_times_one : f (f (f (f 1))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_four_times_one_l1716_171626


namespace NUMINAMATH_CALUDE_quadratic_through_point_l1716_171644

/-- Prove that for a quadratic function y = ax² passing through the point (-1, 4), the value of a is 4. -/
theorem quadratic_through_point (a : ℝ) : (∀ x : ℝ, (a * x^2) = 4) ↔ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_point_l1716_171644


namespace NUMINAMATH_CALUDE_school_problem_solution_l1716_171670

/-- Represents the number of students in each class of a school -/
structure School where
  class1 : ℕ
  class2 : ℕ
  class3 : ℕ
  class4 : ℕ
  class5 : ℕ

/-- The conditions of the school problem -/
def SchoolProblem (s : School) : Prop :=
  s.class1 = 23 ∧
  s.class2 < s.class1 ∧
  s.class3 < s.class2 ∧
  s.class4 < s.class3 ∧
  s.class5 < s.class4 ∧
  s.class1 + s.class2 + s.class3 + s.class4 + s.class5 = 95 ∧
  ∃ (x : ℕ), 
    s.class2 = s.class1 - x ∧
    s.class3 = s.class2 - x ∧
    s.class4 = s.class3 - x ∧
    s.class5 = s.class4 - x

theorem school_problem_solution (s : School) (h : SchoolProblem s) :
  ∃ (x : ℕ), x = 2 ∧
    s.class2 = s.class1 - x ∧
    s.class3 = s.class2 - x ∧
    s.class4 = s.class3 - x ∧
    s.class5 = s.class4 - x :=
  sorry

end NUMINAMATH_CALUDE_school_problem_solution_l1716_171670


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1716_171620

/-- The equation of a line perpendicular to x - 3y + 2 = 0 and passing through (1, -2) -/
theorem perpendicular_line_equation :
  let l₁ : ℝ → ℝ → Prop := λ x y => x - 3 * y + 2 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => 3 * x + y - 1 = 0
  let P : ℝ × ℝ := (1, -2)
  (∀ x y, l₁ x y → (3 * x + y = 0 → False)) ∧ 
  l₂ P.1 P.2 ∧
  ∀ x y, l₂ x y → (x - 3 * y = 0 → False) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1716_171620


namespace NUMINAMATH_CALUDE_special_card_survives_l1716_171629

/-- Represents a deck of cards with a specific card at a given position -/
structure Deck :=
  (size : Nat)
  (special_card_pos : Nat)

/-- Represents a removal operation on the deck -/
inductive Removal
  | left : Nat → Removal
  | right : Nat → Removal

/-- Checks if a card at the given position survives a removal operation -/
def survives (d : Deck) (r : Removal) : Bool :=
  match r with
  | Removal.left n => d.special_card_pos > n
  | Removal.right n => d.size - d.special_card_pos ≥ n

/-- Theorem stating that a card in position 26 or 27 of a 52-card deck can always survive 51 removals -/
theorem special_card_survives (initial_pos : Nat) 
    (h : initial_pos = 26 ∨ initial_pos = 27) : 
    ∀ (removals : List Removal), 
      removals.length = 51 → 
      ∃ (final_deck : Deck), 
        final_deck.size = 1 ∧ 
        final_deck.special_card_pos = 1 :=
  sorry

#check special_card_survives

end NUMINAMATH_CALUDE_special_card_survives_l1716_171629


namespace NUMINAMATH_CALUDE_sin_geq_tan_minus_half_tan_cubed_l1716_171618

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (h : 0 ≤ x ∧ x < Real.pi / 2) :
  Real.sin x ≥ Real.tan x - (1/2) * (Real.tan x)^3 := by
  sorry

end NUMINAMATH_CALUDE_sin_geq_tan_minus_half_tan_cubed_l1716_171618


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l1716_171658

theorem inserted_numbers_sum : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧ 
  (∃ r : ℝ, r > 0 ∧ x = 4 * r ∧ y = 4 * r^2) ∧
  (∃ d : ℝ, y = x + d ∧ 64 = y + d) ∧
  x + y = 131 + 3 * Real.sqrt 129 :=
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l1716_171658


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l1716_171657

theorem consecutive_integers_divisibility (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ n : ℕ, ∃ x y z : ℕ,
    x ∈ Finset.range (2 * c) ∧
    y ∈ Finset.range (2 * c) ∧
    z ∈ Finset.range (2 * c) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (a * b * c) ∣ (x * y * z) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l1716_171657


namespace NUMINAMATH_CALUDE_remaining_fruits_theorem_l1716_171691

/-- Represents the number of fruits in a bag -/
structure FruitBag where
  apples : ℕ
  oranges : ℕ
  mangoes : ℕ

/-- Calculates the total number of fruits in the bag -/
def FruitBag.total (bag : FruitBag) : ℕ :=
  bag.apples + bag.oranges + bag.mangoes

/-- Represents Luisa's actions on the fruit bag -/
def luisa_action (bag : FruitBag) : FruitBag :=
  { apples := bag.apples - 2,
    oranges := bag.oranges - 4,
    mangoes := bag.mangoes - (2 * bag.mangoes / 3) }

/-- The theorem to be proved -/
theorem remaining_fruits_theorem (initial_bag : FruitBag)
    (h1 : initial_bag.apples = 7)
    (h2 : initial_bag.oranges = 8)
    (h3 : initial_bag.mangoes = 15) :
    (luisa_action initial_bag).total = 14 := by
  sorry


end NUMINAMATH_CALUDE_remaining_fruits_theorem_l1716_171691


namespace NUMINAMATH_CALUDE_manolo_face_masks_l1716_171628

/-- Represents the number of face-masks Manolo can make in a given time period -/
def face_masks (first_hour_rate : ℚ) (subsequent_rate : ℚ) (hours : ℚ) : ℚ :=
  let first_hour := min 1 hours
  let remaining_hours := max 0 (hours - 1)
  (60 / first_hour_rate) * first_hour + (60 / subsequent_rate) * remaining_hours

/-- Theorem stating that Manolo makes 45 face-masks in a four-hour shift -/
theorem manolo_face_masks :
  face_masks (4 : ℚ) (6 : ℚ) (4 : ℚ) = 45 := by
  sorry

#eval face_masks (4 : ℚ) (6 : ℚ) (4 : ℚ)

end NUMINAMATH_CALUDE_manolo_face_masks_l1716_171628


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l1716_171636

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l1716_171636


namespace NUMINAMATH_CALUDE_carnation_bouquets_problem_l1716_171676

/-- Proves that given five bouquets of carnations with specified conditions,
    the sum of carnations in the fourth and fifth bouquets is 34. -/
theorem carnation_bouquets_problem (b1 b2 b3 b4 b5 : ℕ) : 
  b1 = 9 → b2 = 14 → b3 = 18 → 
  (b1 + b2 + b3 + b4 + b5) / 5 = 15 →
  b4 + b5 = 34 := by
sorry

end NUMINAMATH_CALUDE_carnation_bouquets_problem_l1716_171676


namespace NUMINAMATH_CALUDE_square_property_implies_equality_l1716_171643

theorem square_property_implies_equality (n : ℕ) (a : ℕ) (a_list : List ℕ) 
  (h : ∀ k : ℕ, ∃ m : ℕ, a * k + 1 = m ^ 2 → 
    ∃ (i : ℕ) (hi : i < a_list.length) (p : ℕ), a_list[i] * k + 1 = p ^ 2) :
  a ∈ a_list := by
  sorry

end NUMINAMATH_CALUDE_square_property_implies_equality_l1716_171643


namespace NUMINAMATH_CALUDE_classroom_contribution_prove_classroom_contribution_l1716_171646

/-- Proves that the amount contributed by each of the eight families is $10 --/
theorem classroom_contribution : ℝ → Prop :=
  fun x =>
    let goal : ℝ := 200
    let raised_from_two : ℝ := 2 * 20
    let raised_from_ten : ℝ := 10 * 5
    let raised_from_eight : ℝ := 8 * x
    let total_raised : ℝ := raised_from_two + raised_from_ten + raised_from_eight
    let remaining : ℝ := 30
    total_raised + remaining = goal → x = 10

/-- Proof of the classroom_contribution theorem --/
theorem prove_classroom_contribution : classroom_contribution 10 := by
  sorry

end NUMINAMATH_CALUDE_classroom_contribution_prove_classroom_contribution_l1716_171646


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l1716_171634

theorem isosceles_triangle_proof (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : 2 * (Real.cos B) * (Real.sin A) = Real.sin C) : A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l1716_171634


namespace NUMINAMATH_CALUDE_linear_system_solution_l1716_171632

theorem linear_system_solution (x y : ℝ) 
  (eq1 : x + 4*y = 5) 
  (eq2 : 5*x + 6*y = 7) : 
  3*x + 5*y = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1716_171632


namespace NUMINAMATH_CALUDE_mail_in_rebates_difference_l1716_171693

/-- The number of additional mail-in rebates compared to bills --/
def additional_rebates : ℕ := 3

/-- The total number of stamps needed --/
def total_stamps : ℕ := 21

/-- The number of thank you cards --/
def thank_you_cards : ℕ := 3

/-- The number of bills --/
def bills : ℕ := 2

theorem mail_in_rebates_difference (rebates : ℕ) (job_applications : ℕ) :
  (thank_you_cards + bills + rebates + job_applications + 1 = total_stamps) →
  (job_applications = 2 * rebates) →
  (rebates = bills + additional_rebates) := by
  sorry

end NUMINAMATH_CALUDE_mail_in_rebates_difference_l1716_171693


namespace NUMINAMATH_CALUDE_two_sixty_billion_scientific_notation_l1716_171633

-- Define 260 billion
def two_hundred_sixty_billion : ℝ := 260000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.6 * (10 ^ 11)

-- Theorem stating that 260 billion is equal to its scientific notation
theorem two_sixty_billion_scientific_notation : 
  two_hundred_sixty_billion = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_two_sixty_billion_scientific_notation_l1716_171633


namespace NUMINAMATH_CALUDE_classroom_gpa_proof_l1716_171660

/-- Proves that the grade point average of one third of a classroom is 30,
    given the grade point average of two thirds is 33 and the overall average is 32. -/
theorem classroom_gpa_proof (gpa_two_thirds : ℝ) (gpa_overall : ℝ) : ℝ :=
  let gpa_one_third : ℝ := 30
  by
    have h1 : gpa_two_thirds = 33 := by sorry
    have h2 : gpa_overall = 32 := by sorry
    have h3 : (1/3 : ℝ) * gpa_one_third + (2/3 : ℝ) * gpa_two_thirds = gpa_overall := by sorry
    sorry

end NUMINAMATH_CALUDE_classroom_gpa_proof_l1716_171660


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1716_171672

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 → 
  L = 1575 → 
  L = 7 * S + R → 
  R = 105 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1716_171672


namespace NUMINAMATH_CALUDE_no_real_roots_l1716_171675

theorem no_real_roots : ∀ x : ℝ, 4 * x^2 - 5 * x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1716_171675


namespace NUMINAMATH_CALUDE_boat_travel_time_l1716_171652

/-- Given a boat that travels 2 miles in 5 minutes, prove that it takes 90 minutes to travel 36 miles at the same speed. -/
theorem boat_travel_time (distance : ℝ) (time : ℝ) (total_distance : ℝ) 
  (h1 : distance = 2) 
  (h2 : time = 5) 
  (h3 : total_distance = 36) : 
  (total_distance / (distance / time)) = 90 := by
  sorry


end NUMINAMATH_CALUDE_boat_travel_time_l1716_171652


namespace NUMINAMATH_CALUDE_jimmy_notebooks_l1716_171653

/-- The number of notebooks Jimmy bought -/
def num_notebooks : ℕ := sorry

/-- The cost of one pen -/
def pen_cost : ℕ := 1

/-- The cost of one notebook -/
def notebook_cost : ℕ := 3

/-- The cost of one folder -/
def folder_cost : ℕ := 5

/-- The number of pens Jimmy bought -/
def num_pens : ℕ := 3

/-- The number of folders Jimmy bought -/
def num_folders : ℕ := 2

/-- The amount Jimmy paid with -/
def paid_amount : ℕ := 50

/-- The amount Jimmy received as change -/
def change_amount : ℕ := 25

theorem jimmy_notebooks :
  num_notebooks = 4 :=
sorry

end NUMINAMATH_CALUDE_jimmy_notebooks_l1716_171653


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l1716_171624

/-- Represents the gross domestic product in billions of yuan -/
def gdp : ℝ := 2502.7

/-- The scientific notation representation of the GDP -/
def scientific_notation : ℝ := 2.5027 * (10 ^ 11)

/-- Theorem stating that the GDP in billions of yuan is equal to its scientific notation representation -/
theorem gdp_scientific_notation_equality : gdp * 10^9 = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_equality_l1716_171624


namespace NUMINAMATH_CALUDE_inequality_proof_l1716_171613

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1716_171613


namespace NUMINAMATH_CALUDE_largest_three_digit_number_with_gcd_condition_l1716_171655

theorem largest_three_digit_number_with_gcd_condition :
  ∃ (x : ℕ), 
    x ≤ 990 ∧ 
    100 ≤ x ∧ 
    x % 3 = 0 ∧
    Nat.gcd 15 (Nat.gcd x 20) = 5 ∧
    ∀ (y : ℕ), 
      100 ≤ y ∧ 
      y ≤ 999 ∧ 
      y % 3 = 0 ∧ 
      Nat.gcd 15 (Nat.gcd y 20) = 5 → 
      y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_with_gcd_condition_l1716_171655


namespace NUMINAMATH_CALUDE_solution_to_equation_l1716_171669

theorem solution_to_equation (z : ℝ) : 
  (z^2 - 5*z + 6)/(z-2) + (5*z^2 + 11*z - 32)/(5*z - 16) = 1 ↔ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1716_171669


namespace NUMINAMATH_CALUDE_distance_right_focus_to_line_l1716_171661

/-- The distance from the right focus of the hyperbola x²/4 - y²/5 = 1 to the line x + 2y - 8 = 0 is √5 -/
theorem distance_right_focus_to_line : ∃ (d : ℝ), d = Real.sqrt 5 ∧ 
  ∀ (x y : ℝ), 
    (x^2 / 4 - y^2 / 5 = 1) →  -- Hyperbola equation
    (x + 2*y - 8 = 0) →       -- Line equation
    d = Real.sqrt ((x - 3)^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_right_focus_to_line_l1716_171661


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1716_171627

/-- The number of players in the basketball team -/
def total_players : ℕ := 16

/-- The number of players to be chosen for a game -/
def team_size : ℕ := 7

/-- The number of players excluding the twins -/
def players_without_twins : ℕ := total_players - 2

/-- The number of ways to choose the team with the given conditions -/
def ways_to_choose_team : ℕ := Nat.choose players_without_twins team_size + Nat.choose players_without_twins (team_size - 2)

theorem basketball_team_selection :
  ways_to_choose_team = 5434 := by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1716_171627


namespace NUMINAMATH_CALUDE_equation_solution_l1716_171630

theorem equation_solution :
  ∃! x : ℝ, (3 / (x - 3) = 4 / (x - 4)) ∧ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1716_171630


namespace NUMINAMATH_CALUDE_tan_sum_17_28_l1716_171614

theorem tan_sum_17_28 : 
  (Real.tan (17 * π / 180) + Real.tan (28 * π / 180)) / 
  (1 - Real.tan (17 * π / 180) * Real.tan (28 * π / 180)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_17_28_l1716_171614


namespace NUMINAMATH_CALUDE_largest_common_divisor_525_385_l1716_171625

theorem largest_common_divisor_525_385 : Nat.gcd 525 385 = 35 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_525_385_l1716_171625


namespace NUMINAMATH_CALUDE_triangle_problem_l1716_171647

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_problem (ABC : Triangle) 
  (h1 : ABC.a * Real.sin ABC.A + ABC.c * Real.sin ABC.C = Real.sqrt 2 * ABC.a * Real.sin ABC.C + ABC.b * Real.sin ABC.B)
  (h2 : ABC.A = 5 * Real.pi / 12) :
  ABC.B = Real.pi / 4 ∧ ABC.a = 1 + Real.sqrt 3 ∧ ABC.c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1716_171647


namespace NUMINAMATH_CALUDE_common_chord_circle_through_AB_center_on_line_smallest_circle_through_AB_l1716_171601

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the line y = -x
def line_y_eq_neg_x (x y : ℝ) : Prop := y = -x

-- Theorem for the common chord
theorem common_chord : ∀ x y : ℝ, C₁ x y ∧ C₂ x y → x - 2*y + 4 = 0 :=
sorry

-- Theorem for the circle passing through A and B with center on y = -x
theorem circle_through_AB_center_on_line : ∃ h k : ℝ, 
  line_y_eq_neg_x h k ∧ 
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = 10 ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) :=
sorry

-- Theorem for the circle with smallest area passing through A and B
theorem smallest_circle_through_AB : ∀ x y : ℝ,
  (x + 2)^2 + (y - 1)^2 = 5 ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) :=
sorry

end NUMINAMATH_CALUDE_common_chord_circle_through_AB_center_on_line_smallest_circle_through_AB_l1716_171601


namespace NUMINAMATH_CALUDE_original_fraction_l1716_171685

theorem original_fraction (x y : ℚ) :
  (x > 0) →
  (y > 0) →
  ((1.2 * x) / (0.75 * y) = 2 / 15) →
  (x / y = 1 / 12) := by
sorry

end NUMINAMATH_CALUDE_original_fraction_l1716_171685


namespace NUMINAMATH_CALUDE_train_distance_l1716_171607

/-- The distance between two trains after 30 seconds -/
theorem train_distance (speed1 speed2 : ℝ) (time : ℝ) : 
  speed1 = 36 →
  speed2 = 48 →
  time = 30 / 3600 →
  let d1 := speed1 * time * 1000
  let d2 := speed2 * time * 1000
  Real.sqrt (d1^2 + d2^2) = 500 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l1716_171607


namespace NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l1716_171695

/-- Proves that Ivy has 20 collectors edition dolls given the conditions -/
theorem ivy_collectors_edition_dolls (dina_dolls ivy_dolls : ℕ) 
  (h1 : dina_dolls = 2 * ivy_dolls)
  (h2 : dina_dolls = 60) :
  (2 : ℚ) / 3 * ivy_dolls = 20 := by
  sorry

end NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l1716_171695


namespace NUMINAMATH_CALUDE_triangle_determinant_l1716_171609

theorem triangle_determinant (A B C : Real) : 
  A + B + C = π → 
  A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2 →
  Matrix.det !![Real.tan A, 1, 1; 1, Real.tan B, 1; 1, 1, Real.tan C] = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_l1716_171609


namespace NUMINAMATH_CALUDE_zero_point_existence_not_necessary_l1716_171664

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

theorem zero_point_existence (a : ℝ) (h : a > 2) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 0, f a x = 0 :=
sorry

theorem not_necessary (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 0, f a x = 0) → a > 2 → False :=
sorry

end NUMINAMATH_CALUDE_zero_point_existence_not_necessary_l1716_171664


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l1716_171665

theorem mean_of_combined_sets (set1_count set1_mean set2_count set2_mean : ℚ) 
  (h1 : set1_count = 4)
  (h2 : set1_mean = 15)
  (h3 : set2_count = 8)
  (h4 : set2_mean = 20) :
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 55 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l1716_171665


namespace NUMINAMATH_CALUDE_problem_1_l1716_171663

theorem problem_1 : 40 + (1/6 - 2/3 + 3/4) * 12 = 43 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1716_171663


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l1716_171697

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

-- Define the theorem
theorem geometric_sequence_a8 (a₁ : ℝ) (q : ℝ) :
  (a₁ * (a₁ * q^2) = 4) →
  (a₁ * q^8 = 256) →
  (geometric_sequence a₁ q 8 = 128 ∨ geometric_sequence a₁ q 8 = -128) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a8_l1716_171697


namespace NUMINAMATH_CALUDE_expression_simplification_l1716_171654

theorem expression_simplification (a b : ℝ) : 
  32 * a^2 * b^2 * (a^2 + b^2)^2 + (a^2 - b^2)^4 + 
  8 * a * b * (a^2 + b^2) * Real.sqrt (16 * a^2 * b^2 * (a^2 + b^2)^2 + (a^2 - b^2)^4) = 
  (a + b)^8 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1716_171654


namespace NUMINAMATH_CALUDE_rectangles_4x4_grid_l1716_171635

/-- The number of rectangles on a 4x4 grid -/
def num_rectangles_4x4 : ℕ :=
  let horizontal_lines := 5
  let vertical_lines := 5
  (horizontal_lines.choose 2) * (vertical_lines.choose 2)

/-- Theorem: The number of rectangles on a 4x4 grid is 100 -/
theorem rectangles_4x4_grid :
  num_rectangles_4x4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_4x4_grid_l1716_171635


namespace NUMINAMATH_CALUDE_students_enjoying_both_music_and_sports_l1716_171615

theorem students_enjoying_both_music_and_sports 
  (total : ℕ) (music : ℕ) (sports : ℕ) (neither : ℕ) : 
  total = 55 → music = 35 → sports = 45 → neither = 4 → 
  music + sports - (total - neither) = 29 := by
sorry

end NUMINAMATH_CALUDE_students_enjoying_both_music_and_sports_l1716_171615


namespace NUMINAMATH_CALUDE_smallest_terminating_with_two_l1716_171639

/-- A function that checks if a positive integer contains the digit 2 -/
def containsDigitTwo (n : ℕ+) : Prop := sorry

/-- A function that checks if the reciprocal of a positive integer is a terminating decimal -/
def isTerminatingDecimal (n : ℕ+) : Prop := sorry

/-- Theorem stating that 2 is the smallest positive integer n such that 1/n is a terminating decimal and n contains the digit 2 -/
theorem smallest_terminating_with_two :
  (∀ m : ℕ+, m < 2 → ¬(isTerminatingDecimal m ∧ containsDigitTwo m)) ∧
  (isTerminatingDecimal 2 ∧ containsDigitTwo 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_terminating_with_two_l1716_171639


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l1716_171674

theorem framed_painting_ratio : 
  ∀ (x : ℝ),
  x > 0 →
  (30 + 2*x) * (20 + 4*x) = 1500 →
  (min (30 + 2*x) (20 + 4*x)) / (max (30 + 2*x) (20 + 4*x)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l1716_171674


namespace NUMINAMATH_CALUDE_mashas_juice_theorem_l1716_171671

/-- Represents Masha's juice drinking process over 3 days -/
def mashas_juice_process (x : ℝ) : Prop :=
  let day1_juice := x - 1
  let day2_juice := (day1_juice^2) / x
  let day3_juice := (day2_juice^2) / x
  let final_juice := (day3_juice^2) / x
  let final_water := x - final_juice
  (final_water = final_juice + 1.5) ∧ (x > 1)

/-- The theorem stating the result of Masha's juice drinking process -/
theorem mashas_juice_theorem :
  ∀ x : ℝ, mashas_juice_process x ↔ (x = 2 ∧ (2 - ((2 - 1)^3) / 2^2 = 1.75)) :=
by sorry

end NUMINAMATH_CALUDE_mashas_juice_theorem_l1716_171671


namespace NUMINAMATH_CALUDE_z_value_l1716_171650

theorem z_value (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x*y - 9) : z = 0 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l1716_171650


namespace NUMINAMATH_CALUDE_candy_sales_l1716_171642

theorem candy_sales (initial_candy : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) :
  initial_candy = 80 →
  sold_monday = 15 →
  remaining_wednesday = 7 →
  initial_candy - sold_monday - remaining_wednesday = 58 := by
  sorry

end NUMINAMATH_CALUDE_candy_sales_l1716_171642


namespace NUMINAMATH_CALUDE_employee_count_l1716_171605

theorem employee_count (avg_salary : ℝ) (new_avg_salary : ℝ) (manager_salary : ℝ) : 
  avg_salary = 1500 →
  new_avg_salary = 2500 →
  manager_salary = 22500 →
  ∃ (E : ℕ), (E : ℝ) * avg_salary + manager_salary = new_avg_salary * ((E : ℝ) + 1) ∧ E = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_count_l1716_171605


namespace NUMINAMATH_CALUDE_normal_price_after_discounts_l1716_171621

theorem normal_price_after_discounts (price : ℝ) : 
  price * (1 - 0.1) * (1 - 0.2) = 144 → price = 200 := by
  sorry

end NUMINAMATH_CALUDE_normal_price_after_discounts_l1716_171621


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l1716_171682

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 - 8*x + y^2 - 6*y + 24 = 0) :
  ∃ (max : ℝ), max = Real.sqrt 5 - 2 ∧ ∀ (x' y' : ℝ), x'^2 - 8*x' + y'^2 - 6*y' + 24 = 0 → x' - 2*y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l1716_171682


namespace NUMINAMATH_CALUDE_fred_final_collection_l1716_171677

/-- Represents the types of coins Fred has --/
inductive Coin
  | Dime
  | Quarter
  | Nickel

/-- Represents Fred's coin collection --/
structure CoinCollection where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

def initial_collection : CoinCollection :=
  { dimes := 7, quarters := 4, nickels := 12 }

def borrowed : CoinCollection :=
  { dimes := 3, quarters := 2, nickels := 0 }

def returned : CoinCollection :=
  { dimes := 0, quarters := 1, nickels := 5 }

def found_cents : ℕ := 50

def cents_per_dime : ℕ := 10

theorem fred_final_collection :
  ∃ (final : CoinCollection),
    final.dimes = 9 ∧
    final.quarters = 3 ∧
    final.nickels = 17 ∧
    final.dimes = initial_collection.dimes - borrowed.dimes + found_cents / cents_per_dime ∧
    final.quarters = initial_collection.quarters - borrowed.quarters + returned.quarters ∧
    final.nickels = initial_collection.nickels + returned.nickels :=
  sorry

end NUMINAMATH_CALUDE_fred_final_collection_l1716_171677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1716_171651

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2017 = 10 →
  a 1 * a 2017 = 16 →
  a 2 + a 1009 + a 2016 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1716_171651


namespace NUMINAMATH_CALUDE_product_power_equals_128y_l1716_171681

theorem product_power_equals_128y (a b : ℤ) (n : ℕ) (h : (a * b) ^ n = 128 * 8) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_power_equals_128y_l1716_171681


namespace NUMINAMATH_CALUDE_vegetarian_eaters_l1716_171683

theorem vegetarian_eaters (only_veg : ℕ) (only_non_veg : ℕ) (both : ℕ) 
  (h1 : only_veg = 15) 
  (h2 : only_non_veg = 8) 
  (h3 : both = 11) : 
  only_veg + both = 26 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_l1716_171683


namespace NUMINAMATH_CALUDE_min_F_beautiful_pair_l1716_171667

def is_beautiful_pair (p q : ℕ) : Prop :=
  ∃ x y : ℕ,
    1 ≤ x ∧ x ≤ 4 ∧
    1 ≤ y ∧ y ≤ 5 ∧
    p = 21 * x + y ∧
    q = 52 + y ∧
    (10 * y + x + 6 * y) % 13 = 0

def F (p q : ℕ) : ℕ :=
  let tens_p := p / 10
  let units_p := p % 10
  let tens_q := q / 10
  let units_q := q % 10
  10 * tens_p + units_q +
  10 * tens_p + units_p +
  10 * units_p + units_q +
  10 * units_p + tens_q

theorem min_F_beautiful_pair :
  ∀ p q : ℕ,
    is_beautiful_pair p q →
    F p q ≥ 156 :=
sorry

end NUMINAMATH_CALUDE_min_F_beautiful_pair_l1716_171667


namespace NUMINAMATH_CALUDE_inequality_proof_l1716_171686

theorem inequality_proof (x y a b : ℝ) (hx : x > 0) (hy : y > 0) :
  ((a * x + b * y) / (x + y))^2 ≤ (a^2 * x + b^2 * y) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1716_171686


namespace NUMINAMATH_CALUDE_scaling_transformation_curve_l1716_171689

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem scaling_transformation_curve (x y x' y' : ℝ) :
  (x' = 5 * x) →
  (y' = 3 * y) →
  (x'^2 + y'^2 = 1) →
  (25 * x^2 + 9 * y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_scaling_transformation_curve_l1716_171689


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l1716_171606

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  (z.im = 0) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l1716_171606


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1716_171608

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 3 ↔ 3 * x + 4 < 5 * x - 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1716_171608


namespace NUMINAMATH_CALUDE_p_true_and_q_false_p_and_not_q_true_l1716_171688

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

-- Theorem stating that p is true and q is false
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

-- Theorem stating that p ∧ ¬q is true
theorem p_and_not_q_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_p_and_not_q_true_l1716_171688


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l1716_171602

-- Define the parabola equation
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 5

-- Define the x-intercept
def x_intercept : ℝ := parabola 0

-- Define the y-intercepts
def y_intercepts : Set ℝ := {y | parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧
  x_intercept + b + c = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l1716_171602


namespace NUMINAMATH_CALUDE_range_of_a_l1716_171698

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ (a < -1 ∨ a > 3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1716_171698


namespace NUMINAMATH_CALUDE_subsets_containing_five_and_six_l1716_171679

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_five_and_six :
  (Finset.filter (λ s : Finset ℕ => 5 ∈ s ∧ 6 ∈ s) (Finset.powerset S)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_subsets_containing_five_and_six_l1716_171679


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l1716_171645

theorem not_p_sufficient_not_necessary_for_q :
  ∃ (x : ℝ), (x > 1 → 1 / x < 1) ∧ (1 / x < 1 → ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l1716_171645


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l1716_171668

/-- Represents a notebook with double-sided pages -/
structure Notebook where
  total_sheets : ℕ
  total_pages : ℕ
  pages_per_sheet : ℕ
  h_pages_per_sheet : pages_per_sheet = 2

/-- Calculates the average of remaining page numbers after borrowing sheets -/
def average_remaining_pages (nb : Notebook) (borrowed_sheets : ℕ) : ℚ :=
  let remaining_pages := nb.total_pages - borrowed_sheets * nb.pages_per_sheet
  let sum_remaining := (nb.total_pages * (nb.total_pages + 1) / 2) -
    (borrowed_sheets * nb.pages_per_sheet * (borrowed_sheets * nb.pages_per_sheet + 1) / 2)
  sum_remaining / remaining_pages

/-- Theorem stating that borrowing 12 sheets results in an average of 23 for remaining pages -/
theorem borrowed_sheets_theorem (nb : Notebook)
    (h_total_sheets : nb.total_sheets = 32)
    (h_total_pages : nb.total_pages = 64)
    (borrowed_sheets : ℕ)
    (h_borrowed : borrowed_sheets = 12) :
    average_remaining_pages nb borrowed_sheets = 23 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l1716_171668


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1716_171680

theorem sum_of_reciprocals_positive 
  (a b c d : ℝ) 
  (ha : |a| > 1) (hb : |b| > 1) (hc : |c| > 1) (hd : |d| > 1)
  (h_eq : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1716_171680


namespace NUMINAMATH_CALUDE_power_mod_thousand_l1716_171666

theorem power_mod_thousand : 7^27 % 1000 = 543 := by sorry

end NUMINAMATH_CALUDE_power_mod_thousand_l1716_171666


namespace NUMINAMATH_CALUDE_polynomial_equality_l1716_171637

def polynomial (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) (x : ℝ) : ℝ :=
  a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8

theorem polynomial_equality 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x - 3)^3 * (2*x + 1)^5 = polynomial a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ x) →
  (a₀ = -27 ∧ a₀ + a₂ + a₄ + a₆ + a₈ = -940) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1716_171637


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l1716_171623

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 6 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l1716_171623


namespace NUMINAMATH_CALUDE_inequality_proof_l1716_171649

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1716_171649


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l1716_171600

/-- Given parametric equations x = √t, y = 2√(1-t), prove they are equivalent to x² + y²/4 = 1, where 0 ≤ x ≤ 1 and 0 ≤ y ≤ 2 -/
theorem parametric_to_standard_equation (t : ℝ) (x y : ℝ) 
    (hx : x = Real.sqrt t) (hy : y = 2 * Real.sqrt (1 - t)) :
    x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l1716_171600


namespace NUMINAMATH_CALUDE_cos_330_degrees_l1716_171631

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l1716_171631


namespace NUMINAMATH_CALUDE_parabola_no_real_roots_l1716_171612

def parabola (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem parabola_no_real_roots :
  ∀ x : ℝ, parabola x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_no_real_roots_l1716_171612


namespace NUMINAMATH_CALUDE_trash_cans_redistribution_l1716_171692

/-- The number of trash cans in Veteran's Park after the redistribution -/
def final_trash_cans_veterans_park (initial_veterans_park : ℕ) : ℕ :=
  let initial_central_park := initial_veterans_park / 2 + 8
  let moved_cans := initial_central_park / 2
  initial_veterans_park + moved_cans

/-- Theorem stating that given 24 initial trash cans in Veteran's Park, 
    the final number of trash cans in Veteran's Park is 34 -/
theorem trash_cans_redistribution :
  final_trash_cans_veterans_park 24 = 34 := by
  sorry

end NUMINAMATH_CALUDE_trash_cans_redistribution_l1716_171692


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1716_171610

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let P : ℝ × ℝ := (-1, 2)
  second_quadrant P.1 P.2 :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1716_171610


namespace NUMINAMATH_CALUDE_tetrahedron_edge_angle_relation_l1716_171678

/-- Theorem about the relationship between opposite edges and angles in a tetrahedron -/
theorem tetrahedron_edge_angle_relation 
  (a a₁ b b₁ c c₁ : ℝ) 
  (α β γ : ℝ) 
  (h_positive : a > 0 ∧ a₁ > 0 ∧ b > 0 ∧ b₁ > 0 ∧ c > 0 ∧ c₁ > 0)
  (h_angles : 0 ≤ α ∧ α ≤ Real.pi / 2 ∧ 0 ≤ β ∧ β ≤ Real.pi / 2 ∧ 0 ≤ γ ∧ γ ≤ Real.pi / 2) :
  (a * a₁ * Real.cos α = b * b₁ * Real.cos β + c * c₁ * Real.cos γ) ∨
  (b * b₁ * Real.cos β = a * a₁ * Real.cos α + c * c₁ * Real.cos γ) ∨
  (c * c₁ * Real.cos γ = a * a₁ * Real.cos α + b * b₁ * Real.cos β) := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_edge_angle_relation_l1716_171678


namespace NUMINAMATH_CALUDE_f_positive_range_f_greater_g_range_l1716_171617

-- Define the functions f and g
def f (x : ℝ) := x^2 - x - 6
def g (b x : ℝ) := b*x - 10

-- Theorem for the range of x where f(x) > 0
theorem f_positive_range (x : ℝ) : 
  f x > 0 ↔ x < -2 ∨ x > 3 :=
sorry

-- Theorem for the range of b where f(x) > g(x) for all real x
theorem f_greater_g_range (b : ℝ) : 
  (∀ x : ℝ, f x > g b x) ↔ b < -5 ∨ b > 3 :=
sorry

end NUMINAMATH_CALUDE_f_positive_range_f_greater_g_range_l1716_171617


namespace NUMINAMATH_CALUDE_exactly_two_true_propositions_l1716_171694

-- Define the propositions
def corresponding_angles_equal : Prop := sorry

def parallel_lines_supplementary_angles : Prop := sorry

def perpendicular_lines_parallel : Prop := sorry

-- Theorem statement
theorem exactly_two_true_propositions :
  (corresponding_angles_equal = false ∧
   parallel_lines_supplementary_angles = true ∧
   perpendicular_lines_parallel = true) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_true_propositions_l1716_171694


namespace NUMINAMATH_CALUDE_problem_statement_l1716_171696

theorem problem_statement : (1 / ((-2^4)^2)) * ((-2)^7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1716_171696


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_equality_l1716_171604

theorem sqrt_18_minus_sqrt_2_equality (a b : ℝ) :
  Real.sqrt 18 - Real.sqrt 2 = a * Real.sqrt 2 - Real.sqrt 2 ∧
  a * Real.sqrt 2 - Real.sqrt 2 = b * Real.sqrt 2 →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_equality_l1716_171604


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1716_171641

theorem arithmetic_expression_equality : 10 - 9 + 8 * (7 - 6) + 5 * 4 - 3 + 2 - 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1716_171641
