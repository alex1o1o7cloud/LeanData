import Mathlib

namespace pirate_captain_age_l3196_319691

/-- Represents a health insurance card number -/
structure HealthInsuranceCard where
  main_number : Nat
  control_number : Nat
  h_main_digits : main_number < 10000000000000
  h_control_digits : control_number < 100

/-- Checks if a health insurance card number is valid -/
def is_valid_card (card : HealthInsuranceCard) : Prop :=
  (card.main_number + card.control_number) % 97 = 0

/-- Calculates the age based on birth year and current year -/
def calculate_age (birth_year : Nat) (current_year : Nat) : Nat :=
  current_year - birth_year

theorem pirate_captain_age :
  ∃ (card : HealthInsuranceCard),
    card.control_number = 67 ∧
    ∃ (x : Nat), x < 10 ∧ card.main_number = 1000000000000 * (10 + x) + 1271153044 ∧
    is_valid_card card ∧
    calculate_age (1900 + (10 + x)) 2011 = 65 := by
  sorry

end pirate_captain_age_l3196_319691


namespace machinery_spending_l3196_319653

/-- Represents the financial breakdown of Kanul's spending --/
structure KanulSpending where
  total : ℝ
  rawMaterials : ℝ
  cash : ℝ
  machinery : ℝ

/-- Theorem stating the amount spent on machinery --/
theorem machinery_spending (k : KanulSpending) 
  (h1 : k.total = 1000)
  (h2 : k.rawMaterials = 500)
  (h3 : k.cash = 0.1 * k.total)
  (h4 : k.total = k.rawMaterials + k.cash + k.machinery) :
  k.machinery = 400 := by
  sorry

end machinery_spending_l3196_319653


namespace sum_remainder_mod_20_l3196_319607

theorem sum_remainder_mod_20 : (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 20 = 18 := by
  sorry

end sum_remainder_mod_20_l3196_319607


namespace algebraic_inequalities_l3196_319630

theorem algebraic_inequalities :
  (∀ a : ℝ, a^2 + 2 > 2*a) ∧
  (∀ x : ℝ, (x+5)*(x+7) < (x+6)^2) := by
  sorry

end algebraic_inequalities_l3196_319630


namespace inverse_p_is_true_l3196_319620

-- Define the original proposition
def p (x : ℝ) : Prop := x < -3 → x^2 - 2*x - 8 > 0

-- Define the inverse of the proposition
def p_inverse (x : ℝ) : Prop := ¬(x < -3) → ¬(x^2 - 2*x - 8 > 0)

-- Theorem stating that the inverse of p is true
theorem inverse_p_is_true : ∀ x : ℝ, p_inverse x :=
  sorry

end inverse_p_is_true_l3196_319620


namespace anoop_investment_l3196_319666

/-- Calculates the investment amount of the second partner in a business partnership --/
def calculate_second_partner_investment (first_partner_investment : ℕ) (first_partner_months : ℕ) (second_partner_months : ℕ) : ℕ :=
  (first_partner_investment * first_partner_months) / second_partner_months

/-- Proves that Anoop's investment is 40,000 given the problem conditions --/
theorem anoop_investment :
  let arjun_investment : ℕ := 20000
  let total_months : ℕ := 12
  let anoop_months : ℕ := 6
  calculate_second_partner_investment arjun_investment total_months anoop_months = 40000 := by
  sorry

#eval calculate_second_partner_investment 20000 12 6

end anoop_investment_l3196_319666


namespace inscribed_square_area_l3196_319633

/-- Given a circle with equation 2x^2 = -2y^2 + 16x - 8y + 40, 
    the area of a square inscribed around it with one pair of sides 
    parallel to the x-axis is 160 square units. -/
theorem inscribed_square_area (x y : ℝ) : 
  2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 40 → 
  ∃ (s : ℝ), s > 0 ∧ s^2 = 160 ∧ 
  ∃ (cx cy : ℝ), (x - cx)^2 + (y - cy)^2 ≤ (s/2)^2 :=
sorry

end inscribed_square_area_l3196_319633


namespace hotel_assignment_problem_l3196_319655

/-- The number of ways to assign friends to rooms -/
def assignFriendsToRooms (numFriends numRooms maxPerRoom : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of assignments for the given problem -/
theorem hotel_assignment_problem :
  assignFriendsToRooms 6 5 2 = 7200 := by
  sorry

end hotel_assignment_problem_l3196_319655


namespace circle_C_equation_max_y_over_x_l3196_319603

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line that intersects with x-axis to form the center of circle C
def center_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line tangent to circle C
def tangent_line (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle for the second part of the problem
def circle_P (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Theorem for the first part of the problem
theorem circle_C_equation :
  ∀ x y : ℝ, 
  (∃ x₀, center_line x₀ 0 ∧ (∀ x y, circle_C x y → (x - x₀)^2 + y^2 = 2)) →
  (∃ d : ℝ, d > 0 ∧ ∀ x y, circle_C x y → d = |x + y + 3| / Real.sqrt 2) →
  circle_C x y ↔ (x + 1)^2 + y^2 = 2 :=
sorry

-- Theorem for the second part of the problem
theorem max_y_over_x :
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
   ∀ x y : ℝ, circle_P x y → |y / x| ≤ k ∧ 
   ∃ x₀ y₀ : ℝ, circle_P x₀ y₀ ∧ |y₀ / x₀| = k) :=
sorry

end circle_C_equation_max_y_over_x_l3196_319603


namespace rich_walk_distance_l3196_319676

-- Define the walking pattern
def house_to_sidewalk : ℕ := 20
def sidewalk_to_road_end : ℕ := 200
def left_turn_multiplier : ℕ := 2
def final_stretch_divisor : ℕ := 2

-- Define the total distance walked
def total_distance : ℕ :=
  let initial_distance := house_to_sidewalk + sidewalk_to_road_end
  let after_left_turn := initial_distance + left_turn_multiplier * initial_distance
  let to_end_point := after_left_turn + after_left_turn / final_stretch_divisor
  2 * to_end_point

-- Theorem statement
theorem rich_walk_distance : total_distance = 1980 := by sorry

end rich_walk_distance_l3196_319676


namespace sum_of_fractions_l3196_319684

theorem sum_of_fractions : (1 : ℚ) / 1 + (2 : ℚ) / 2 + (3 : ℚ) / 3 = 3 := by
  sorry

end sum_of_fractions_l3196_319684


namespace theater_ticket_sales_l3196_319694

theorem theater_ticket_sales (orchestra_price balcony_price premium_price : ℕ)
                             (total_tickets : ℕ) (total_revenue : ℕ)
                             (orchestra balcony premium : ℕ) :
  orchestra_price = 15 →
  balcony_price = 10 →
  premium_price = 25 →
  total_tickets = 550 →
  total_revenue = 9750 →
  orchestra + balcony + premium = total_tickets →
  orchestra_price * orchestra + balcony_price * balcony + premium_price * premium = total_revenue →
  premium = 5 * orchestra →
  orchestra ≥ 50 →
  balcony - orchestra = 179 :=
by sorry

end theater_ticket_sales_l3196_319694


namespace two_books_into_five_l3196_319671

/-- The number of ways to insert new books into a shelf while maintaining the order of existing books -/
def insert_books (original : ℕ) (new : ℕ) : ℕ :=
  (original + 1) * (original + 2) / 2

/-- Theorem stating that inserting 2 books into a shelf with 5 books results in 42 different arrangements -/
theorem two_books_into_five : insert_books 5 2 = 42 := by
  sorry

end two_books_into_five_l3196_319671


namespace book_purchasing_problem_l3196_319640

/-- Represents a book purchasing plan. -/
structure BookPlan where
  classics : ℕ
  comics : ℕ

/-- Checks if a book plan is valid according to the given conditions. -/
def isValidPlan (p : BookPlan) (classicPrice comicPrice : ℕ) : Prop :=
  p.comics = p.classics + 20 ∧
  p.classics + p.comics ≥ 72 ∧
  classicPrice * p.classics + comicPrice * p.comics ≤ 2000

theorem book_purchasing_problem :
  ∃ (classicPrice comicPrice : ℕ),
    -- Given conditions
    20 * classicPrice + 40 * comicPrice = 1520 ∧
    20 * classicPrice - 20 * comicPrice = 440 ∧
    -- Prove the following
    classicPrice = 40 ∧
    comicPrice = 18 ∧
    (∀ p : BookPlan, isValidPlan p classicPrice comicPrice →
      (p.classics = 26 ∧ p.comics = 46) ∨
      (p.classics = 27 ∧ p.comics = 47) ∨
      (p.classics = 28 ∧ p.comics = 48)) ∧
    (∀ c : ℕ, c ∈ [26, 27, 28] →
      isValidPlan ⟨c, c + 20⟩ classicPrice comicPrice) :=
by sorry

end book_purchasing_problem_l3196_319640


namespace apple_orange_pricing_l3196_319670

/-- The price of an orange in dollars -/
def orange_price : ℝ := 2

/-- The price of an apple in dollars -/
def apple_price : ℝ := 3 * orange_price

theorem apple_orange_pricing :
  (4 * apple_price + 7 * orange_price = 38) →
  (orange_price = 2 ∧ 5 * apple_price = 30) := by
  sorry

end apple_orange_pricing_l3196_319670


namespace inequality_of_distinct_positives_l3196_319641

theorem inequality_of_distinct_positives (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 := by
  sorry

end inequality_of_distinct_positives_l3196_319641


namespace A_solution_l3196_319662

noncomputable def A (x y : ℝ) : ℝ := 
  (Real.sqrt (4 * (x - Real.sqrt y) + y / x) * 
   Real.sqrt (9 * x^2 + 6 * (2 * y * x^3)^(1/3) + (4 * y^2)^(1/3))) / 
  (6 * x^2 + 2 * (2 * y * x^3)^(1/3) - 3 * Real.sqrt (y * x^2) - (4 * y^5)^(1/6)) / 2.343

theorem A_solution (x y : ℝ) (hx : x > 0) (hy : y ≥ 0) :
  A x y = if y > 4 * x^2 then -1 / Real.sqrt x else 1 / Real.sqrt x :=
by sorry

end A_solution_l3196_319662


namespace bad_shape_cards_l3196_319649

/-- Calculates the number of baseball cards in bad shape given the initial conditions and distributions --/
theorem bad_shape_cards (initial : ℕ) (from_father : ℕ) (from_ebay : ℕ) (to_dexter : ℕ) (kept : ℕ) : 
  initial + from_father + from_ebay - (to_dexter + kept) = 4 :=
by
  sorry

#check bad_shape_cards 4 13 36 29 20

end bad_shape_cards_l3196_319649


namespace luke_birthday_stickers_l3196_319692

/-- Represents the number of stickers Luke has at different stages --/
structure StickerCount where
  initial : ℕ
  bought : ℕ
  birthday : ℕ
  given_away : ℕ
  used : ℕ
  final : ℕ

/-- Calculates the number of stickers Luke got for his birthday --/
def birthday_stickers (s : StickerCount) : ℕ :=
  s.final + s.given_away + s.used - s.initial - s.bought

/-- Theorem stating that Luke got 20 stickers for his birthday --/
theorem luke_birthday_stickers :
  ∀ s : StickerCount,
    s.initial = 20 ∧
    s.bought = 12 ∧
    s.given_away = 5 ∧
    s.used = 8 ∧
    s.final = 39 →
    birthday_stickers s = 20 := by
  sorry


end luke_birthday_stickers_l3196_319692


namespace exists_double_area_quadrilateral_l3196_319609

/-- The area of a quadrilateral given by four points in the plane -/
noncomputable def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the existence of points A, B, C, and D such that 
    the area of ABCD is twice the area of ADBC -/
theorem exists_double_area_quadrilateral :
  ∃ (A B C D : ℝ × ℝ), quadrilateralArea A B C D = 2 * quadrilateralArea A D B C := by
  sorry

end exists_double_area_quadrilateral_l3196_319609


namespace map_scale_conversion_l3196_319664

/-- Given a map scale where 10 cm represents 50 km, 
    prove that a 23 cm length on the map represents 115 km. -/
theorem map_scale_conversion (scale : ℝ → ℝ) : 
  (scale 10 = 50) → (scale 23 = 115) :=
by
  sorry

end map_scale_conversion_l3196_319664


namespace plant_species_numbering_not_unique_l3196_319627

theorem plant_species_numbering_not_unique : ∃ a b : ℕ, 
  2 ≤ a ∧ a < b ∧ b ≤ 20000 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 20000 → Nat.gcd a k = Nat.gcd b k) :=
sorry

end plant_species_numbering_not_unique_l3196_319627


namespace sandra_amount_sandra_gets_100_l3196_319647

def share_money (sandra_ratio : ℕ) (amy_ratio : ℕ) (ruth_ratio : ℕ) (amy_amount : ℕ) : ℕ → ℕ → ℕ → Prop :=
  λ sandra_amount ruth_amount total_amount =>
    sandra_amount * amy_ratio = amy_amount * sandra_ratio ∧
    ruth_amount * amy_ratio = amy_amount * ruth_ratio ∧
    total_amount = sandra_amount + amy_amount + ruth_amount

theorem sandra_amount (amy_amount : ℕ) :
  share_money 2 1 3 amy_amount (2 * amy_amount) (3 * amy_amount) (6 * amy_amount) :=
by sorry

theorem sandra_gets_100 :
  share_money 2 1 3 50 100 150 300 :=
by sorry

end sandra_amount_sandra_gets_100_l3196_319647


namespace M_intersect_N_l3196_319688

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem M_intersect_N : M ∩ N = {2, 4} := by sorry

end M_intersect_N_l3196_319688


namespace movie_ticket_sales_l3196_319663

theorem movie_ticket_sales (adult_price student_price total_revenue : ℚ)
  (student_tickets : ℕ) (h1 : adult_price = 4)
  (h2 : student_price = 5 / 2) (h3 : total_revenue = 445 / 2)
  (h4 : student_tickets = 9) :
  ∃ (adult_tickets : ℕ),
    adult_price * adult_tickets + student_price * student_tickets = total_revenue ∧
    adult_tickets + student_tickets = 59 := by
  sorry

end movie_ticket_sales_l3196_319663


namespace sentence_has_32_letters_l3196_319660

def original_sentence : String := "В ЭТОМ ПРЕДЛОЖЕНИИ ... БУКВ"
def filled_word : String := "ТРИДЦАТЬ ДВЕ"
def full_sentence : String := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

def is_cyrillic_letter (c : Char) : Bool :=
  (c.toNat ≥ 1040 ∧ c.toNat ≤ 1103) ∨ (c = 'Ё' ∨ c = 'ё')

def count_cyrillic_letters (s : String) : Nat :=
  s.toList.filter is_cyrillic_letter |>.length

theorem sentence_has_32_letters : count_cyrillic_letters full_sentence = 32 := by
  sorry

end sentence_has_32_letters_l3196_319660


namespace no_valid_sequence_for_certain_n_l3196_319658

/-- A sequence where each number from 1 to n appears twice, 
    and the second occurrence of each number r is r positions after its first occurrence -/
def ValidSequence (n : ℕ) (seq : List ℕ) : Prop :=
  (seq.length = 2 * n) ∧
  (∀ r ∈ Finset.range n, 
    ∃ i j, seq.nthLe i (by sorry) = r + 1 ∧ 
           seq.nthLe j (by sorry) = r + 1 ∧ 
           j = i + (r + 1))

theorem no_valid_sequence_for_certain_n (n : ℕ) :
  (∃ seq : List ℕ, ValidSequence n seq) → 
  (n % 4 ≠ 2 ∧ n % 4 ≠ 3) :=
by sorry

end no_valid_sequence_for_certain_n_l3196_319658


namespace cost_per_set_is_20_l3196_319696

/-- Represents the manufacturing and sales scenario for horseshoe sets -/
structure HorseshoeManufacturing where
  initialOutlay : ℕ
  sellingPrice : ℕ
  setsSold : ℕ
  profit : ℕ

/-- Calculates the cost per set given the manufacturing scenario -/
def costPerSet (h : HorseshoeManufacturing) : ℚ :=
  (h.sellingPrice * h.setsSold - h.profit - h.initialOutlay) / h.setsSold

/-- Theorem stating that the cost per set is $20 given the specific scenario -/
theorem cost_per_set_is_20 (h : HorseshoeManufacturing) 
  (h_initial : h.initialOutlay = 10000)
  (h_price : h.sellingPrice = 50)
  (h_sold : h.setsSold = 500)
  (h_profit : h.profit = 5000) :
  costPerSet h = 20 := by
  sorry

#eval costPerSet { initialOutlay := 10000, sellingPrice := 50, setsSold := 500, profit := 5000 }

end cost_per_set_is_20_l3196_319696


namespace systematic_sample_fourth_element_exists_and_unique_l3196_319637

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem systematic_sample_fourth_element_exists_and_unique
  (total_students : ℕ)
  (sample_size : ℕ)
  (n1 n2 n3 : ℕ)
  (h_total : total_students = 52)
  (h_sample : sample_size = 4)
  (h_n1 : n1 = 6)
  (h_n2 : n2 = 32)
  (h_n3 : n3 = 45)
  (h_distinct : n1 < n2 ∧ n2 < n3)
  (h_valid : n1 ≤ total_students ∧ n2 ≤ total_students ∧ n3 ≤ total_students) :
  ∃! n4 : ℕ,
    ∃ s : SystematicSample,
      s.population_size = total_students ∧
      s.sample_size = sample_size ∧
      isInSample s n1 ∧
      isInSample s n2 ∧
      isInSample s n3 ∧
      isInSample s n4 ∧
      n4 ≠ n1 ∧ n4 ≠ n2 ∧ n4 ≠ n3 :=
by sorry

end systematic_sample_fourth_element_exists_and_unique_l3196_319637


namespace quadratic_roots_sum_product_l3196_319656

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (m^2 - 4*m - 1 = 0) → 
  (n^2 - 4*n - 1 = 0) → 
  (m + n = 4) → 
  (m * n = -1) → 
  m + n - m * n = 5 := by
sorry

end quadratic_roots_sum_product_l3196_319656


namespace functional_equation_solution_l3196_319681

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x + y) = f x + f y - 2023) : 
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 := by
  sorry

end functional_equation_solution_l3196_319681


namespace only_B_is_true_l3196_319679

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Proposition A
def propA (P₀ : Point2D) (l : Line2D) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ y - P₀.y = k * (x - P₀.x)

-- Proposition B
def propB (P₁ P₂ : Point2D) (l : Line2D) : Prop :=
  P₁ ≠ P₂ → ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ 
    (y - P₁.y) * (P₂.x - P₁.x) = (x - P₁.x) * (P₂.y - P₁.y)

-- Proposition C
def propC (l : Line2D) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ x / a + y / b = 1

-- Proposition D
def propD (b : ℝ) (l : Line2D) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, l.a * x + l.b * y + l.c = 0 ↔ y = k * x + b

theorem only_B_is_true :
  (∃ P₀ : Point2D, ∀ l : Line2D, propA P₀ l) = false ∧
  (∀ P₁ P₂ : Point2D, ∀ l : Line2D, propB P₁ P₂ l) = true ∧
  (∀ l : Line2D, propC l) = false ∧
  (∃ b : ℝ, ∀ l : Line2D, propD b l) = false :=
sorry

end only_B_is_true_l3196_319679


namespace repeating_decimal_to_fraction_l3196_319648

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 127 / 999) ∧ (x = 3124 / 999) := by
  sorry

end repeating_decimal_to_fraction_l3196_319648


namespace kim_shirts_fraction_l3196_319674

theorem kim_shirts_fraction (initial_shirts : ℕ) (remaining_shirts : ℕ) :
  initial_shirts = 4 * 12 →
  remaining_shirts = 32 →
  (initial_shirts - remaining_shirts : ℚ) / initial_shirts = 1 / 3 := by
  sorry

end kim_shirts_fraction_l3196_319674


namespace smallest_y_for_divisibility_by_11_l3196_319629

/-- Given a number in the form 7y86038 where y is a single digit (0-9),
    2 is the smallest whole number for y that makes the number divisible by 11. -/
theorem smallest_y_for_divisibility_by_11 :
  ∃ (y : ℕ), y ≤ 9 ∧ 
  (7 * 10^6 + y * 10^5 + 8 * 10^4 + 6 * 10^3 + 0 * 10^2 + 3 * 10 + 8) % 11 = 0 ∧
  ∀ (z : ℕ), z < y → (7 * 10^6 + z * 10^5 + 8 * 10^4 + 6 * 10^3 + 0 * 10^2 + 3 * 10 + 8) % 11 ≠ 0 ∧
  y = 2 :=
by sorry

end smallest_y_for_divisibility_by_11_l3196_319629


namespace shaded_fraction_of_rectangle_l3196_319628

theorem shaded_fraction_of_rectangle (length width : ℝ) (h1 : length = 10) (h2 : width = 15) :
  let total_area := length * width
  let third_area := total_area / 3
  let shaded_area := third_area / 2
  shaded_area / total_area = 1 / 6 := by
  sorry

end shaded_fraction_of_rectangle_l3196_319628


namespace coefficient_of_x_cubed_l3196_319657

def expression (x : ℝ) : ℝ :=
  5 * (x^2 - 2*x^3 + x) + 2 * (x + 3*x^3 - 4*x^2 + 2*x^5 + 2*x^3) - 7 * (2 + x - 5*x^3 - 2*x^2)

theorem coefficient_of_x_cubed (x : ℝ) :
  ∃ (a b c d : ℝ), expression x = a*x^5 + b*x^4 + 35*x^3 + c*x^2 + d*x + (5*1 - 7*2) :=
by sorry

end coefficient_of_x_cubed_l3196_319657


namespace horner_v4_value_l3196_319651

/-- The polynomial f(x) = 12 + 35x - 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

/-- The fourth intermediate value in Horner's method for polynomial f -/
def v4 (x : ℝ) : ℝ := (((3*x + 5)*x + 6)*x + 79)*x - 8

/-- Theorem: The value of v4 for f(x) at x = -4 is 220 -/
theorem horner_v4_value : v4 (-4) = 220 := by sorry

end horner_v4_value_l3196_319651


namespace total_teaching_years_is_70_l3196_319654

/-- The total number of years Tom and Devin have been teaching -/
def total_teaching_years (tom_years devin_years : ℕ) : ℕ := tom_years + devin_years

/-- Tom's teaching years -/
def tom_years : ℕ := 50

/-- Devin's teaching years in terms of Tom's -/
def devin_years : ℕ := tom_years / 2 - 5

theorem total_teaching_years_is_70 : 
  total_teaching_years tom_years devin_years = 70 := by sorry

end total_teaching_years_is_70_l3196_319654


namespace parabola_point_ordering_l3196_319673

def f (x : ℝ) : ℝ := -x^2 + 5

theorem parabola_point_ordering :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-4) = y₁ ∧ f (-1) = y₂ ∧ f 2 = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ :=
by sorry

end parabola_point_ordering_l3196_319673


namespace oranges_from_ann_l3196_319622

theorem oranges_from_ann (initial_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 9)
  (h2 : final_oranges = 38) :
  final_oranges - initial_oranges = 29 := by
  sorry

end oranges_from_ann_l3196_319622


namespace circle_radius_is_six_l3196_319669

theorem circle_radius_is_six (r : ℝ) (h : r > 0) :
  2 * 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → r = 6 := by
  sorry

end circle_radius_is_six_l3196_319669


namespace horse_journey_l3196_319616

/-- Given a geometric sequence with common ratio 1/2 and sum of first 7 terms equal to 700,
    the sum of the first 14 terms is 22575/32 -/
theorem horse_journey (a : ℝ) (S : ℕ → ℝ) : 
  (∀ n, S (n + 1) = S n + a * (1/2)^n) → 
  S 0 = 0 →
  S 7 = 700 →
  S 14 = 22575/32 := by
sorry

end horse_journey_l3196_319616


namespace math_books_count_l3196_319636

/-- Proves that the number of math books bought is 27 given the conditions of the problem -/
theorem math_books_count (total_books : ℕ) (math_book_price history_book_price total_price : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_book_price = 4)
  (h3 : history_book_price = 5)
  (h4 : total_price = 373) :
  ∃ (math_books : ℕ), 
    math_books + (total_books - math_books) = total_books ∧ 
    math_books * math_book_price + (total_books - math_books) * history_book_price = total_price ∧
    math_books = 27 :=
by
  sorry

end math_books_count_l3196_319636


namespace max_piece_length_and_total_pieces_l3196_319626

-- Define the lengths of the two pipes
def pipe1_length : ℕ := 42
def pipe2_length : ℕ := 63

-- Define the theorem
theorem max_piece_length_and_total_pieces :
  ∃ (max_length : ℕ) (total_pieces : ℕ),
    max_length = Nat.gcd pipe1_length pipe2_length ∧
    max_length = 21 ∧
    total_pieces = pipe1_length / max_length + pipe2_length / max_length ∧
    total_pieces = 5 := by
  sorry

end max_piece_length_and_total_pieces_l3196_319626


namespace circle_equation_l3196_319665

/-- Prove that the equation (x-1)^2 + (y-1)^2 = 2 represents the circle with center (1,1) passing through the point (2,2). -/
theorem circle_equation (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ - 1)^2 + (y₀ - 1)^2 = 2 ↔ 
    ((x₀ - 1)^2 + (y₀ - 1)^2 = (x - 1)^2 + (y - 1)^2 ∧ (x - 1)^2 + (y - 1)^2 = 1)) ∧
  (2 - 1)^2 + (2 - 1)^2 = 2 := by
sorry


end circle_equation_l3196_319665


namespace sqrt_two_function_value_l3196_319631

/-- Given a function f where f(x-1) = x^2 - 2x for all real x, prove that f(√2) = 1 -/
theorem sqrt_two_function_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1) = x^2 - 2*x) : 
  f (Real.sqrt 2) = 1 := by
  sorry

end sqrt_two_function_value_l3196_319631


namespace inequality_solution_set_l3196_319639

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (x - 2) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by
  sorry

end inequality_solution_set_l3196_319639


namespace joan_sofa_cost_l3196_319618

theorem joan_sofa_cost (joan_cost karl_cost : ℝ) 
  (sum_condition : joan_cost + karl_cost = 600)
  (price_relation : 2 * joan_cost = karl_cost + 90) : 
  joan_cost = 230 := by
sorry

end joan_sofa_cost_l3196_319618


namespace equivalent_angle_proof_l3196_319693

/-- The angle (in degrees) that has the same terminal side as -60° within [0°, 360°) -/
def equivalent_angle : ℝ := 300

theorem equivalent_angle_proof :
  ∃ (k : ℤ), equivalent_angle = k * 360 - 60 ∧ 
  0 ≤ equivalent_angle ∧ equivalent_angle < 360 :=
by sorry

end equivalent_angle_proof_l3196_319693


namespace probability_factor_less_than_7_l3196_319646

def factors_of_72 : Finset ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def factors_less_than_7 : Finset ℕ := {1, 2, 3, 4, 6}

theorem probability_factor_less_than_7 :
  (factors_less_than_7.card : ℚ) / (factors_of_72.card : ℚ) = 5 / 12 := by sorry

end probability_factor_less_than_7_l3196_319646


namespace chinese_remainder_theorem_example_l3196_319682

theorem chinese_remainder_theorem_example :
  ∃ x : ℤ, (x ≡ 1 [ZMOD 3] ∧
             x ≡ -1 [ZMOD 5] ∧
             x ≡ 2 [ZMOD 7] ∧
             x ≡ -2 [ZMOD 11]) ↔
            x ≡ 394 [ZMOD 1155] := by
  sorry

end chinese_remainder_theorem_example_l3196_319682


namespace exponential_inequality_l3196_319698

theorem exponential_inequality (a b c : ℝ) :
  0 < 0.8 ∧ 0.8 < 1 ∧ 5.2 > 1 →
  0.8^5.5 < 0.8^5.2 ∧ 0.8^5.2 < 5.2^0.1 :=
by sorry

end exponential_inequality_l3196_319698


namespace intersection_implies_value_l3196_319687

theorem intersection_implies_value (a : ℝ) : 
  let A : Set ℝ := {2, a - 1}
  let B : Set ℝ := {a^2 - 7, -1}
  A ∩ B = {2} → a = -3 :=
by
  sorry

end intersection_implies_value_l3196_319687


namespace smallest_w_sum_of_digits_17_l3196_319685

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest positive integer w such that 10^w - 74 has a sum of digits equal to 17 is 3 -/
theorem smallest_w_sum_of_digits_17 :
  ∀ w : ℕ+, sum_of_digits (10^(w.val) - 74) = 17 → w.val ≥ 3 :=
by sorry

end smallest_w_sum_of_digits_17_l3196_319685


namespace janes_age_l3196_319601

theorem janes_age (agnes_age : ℕ) (future_years : ℕ) (jane_age : ℕ) : 
  agnes_age = 25 → 
  future_years = 13 → 
  agnes_age + future_years = 2 * (jane_age + future_years) → 
  jane_age = 6 := by
sorry

end janes_age_l3196_319601


namespace bruce_payment_l3196_319613

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grapes_quantity grapes_rate mangoes_quantity mangoes_rate : ℕ) : ℕ :=
  grapes_quantity * grapes_rate + mangoes_quantity * mangoes_rate

/-- Theorem stating that Bruce paid 1055 to the shopkeeper -/
theorem bruce_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end bruce_payment_l3196_319613


namespace inequality_system_solution_l3196_319661

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (2 * x - a < 0 ∧ 1 - 2 * x ≥ 7) ↔ x ≤ -3) → 
  a > -6 :=
by sorry

end inequality_system_solution_l3196_319661


namespace batman_game_cost_l3196_319634

def football_cost : ℚ := 14.02
def strategy_cost : ℚ := 9.46
def total_spent : ℚ := 35.52

theorem batman_game_cost :
  ∃ (batman_cost : ℚ),
    batman_cost = total_spent - football_cost - strategy_cost ∧
    batman_cost = 12.04 :=
by sorry

end batman_game_cost_l3196_319634


namespace imaginary_part_of_z_l3196_319683

theorem imaginary_part_of_z (z : ℂ) (h : (2 + Complex.I) * z = 2 - 4 * Complex.I) : 
  Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l3196_319683


namespace max_intersections_ellipse_cosine_l3196_319624

-- Define the ellipse equation
def ellipse (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

-- Define the cosine function
def cosine_graph (x y : ℝ) : Prop :=
  y = Real.cos x

-- Theorem statement
theorem max_intersections_ellipse_cosine :
  ∃ (h k a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∃ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → ellipse p.1 p.2 h k a b ∧ cosine_graph p.1 p.2) ∧
    points.card = 8) ∧
  (∀ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → ellipse p.1 p.2 h k a b ∧ cosine_graph p.1 p.2) →
    points.card ≤ 8) :=
by sorry


end max_intersections_ellipse_cosine_l3196_319624


namespace true_propositions_l3196_319644

-- Define the propositions
variable (p₁ p₂ p₃ p₄ : Prop)

-- Define the truth values of the propositions
axiom p₁_true : p₁
axiom p₂_false : ¬p₂
axiom p₃_false : ¬p₃
axiom p₄_true : p₄

-- Define the compound propositions
def prop1 := p₁ ∧ p₄
def prop2 := p₁ ∧ p₂
def prop3 := ¬p₂ ∨ p₃
def prop4 := ¬p₃ ∨ ¬p₄

-- Theorem to prove
theorem true_propositions : 
  prop1 p₁ p₄ ∧ prop3 p₂ p₃ ∧ prop4 p₃ p₄ ∧ ¬(prop2 p₁ p₂) :=
sorry

end true_propositions_l3196_319644


namespace count_four_digit_integers_l3196_319650

theorem count_four_digit_integers (y : ℕ) : 
  (∃ (n : ℕ), 1000 ≤ y ∧ y < 10000 ∧ (5678 * y + 123) % 29 = 890 % 29) →
  (Finset.filter (λ y => 1000 ≤ y ∧ y < 10000 ∧ (5678 * y + 123) % 29 = 890 % 29) (Finset.range 10000)).card = 310 :=
by sorry

end count_four_digit_integers_l3196_319650


namespace unique_real_root_of_polynomial_l3196_319623

theorem unique_real_root_of_polynomial (x : ℝ) :
  x^4 - 4*x^3 + 5*x^2 - 2*x + 2 = 0 ↔ x = 1 := by
  sorry

end unique_real_root_of_polynomial_l3196_319623


namespace dumpling_storage_temp_l3196_319652

def storage_temp_range (x : ℝ) : Prop := -20 ≤ x ∧ x ≤ -16

theorem dumpling_storage_temp :
  (storage_temp_range (-17)) ∧
  (storage_temp_range (-18)) ∧
  (storage_temp_range (-19)) ∧
  (¬ storage_temp_range (-22)) :=
by sorry

end dumpling_storage_temp_l3196_319652


namespace candy_bar_cost_l3196_319625

theorem candy_bar_cost (initial_amount : ℝ) (num_candy_bars : ℕ) (remaining_amount : ℝ) :
  initial_amount = 20 →
  num_candy_bars = 4 →
  remaining_amount = 12 →
  (initial_amount - remaining_amount) / num_candy_bars = 2 :=
by sorry

end candy_bar_cost_l3196_319625


namespace cubic_polynomials_common_roots_l3196_319695

theorem cubic_polynomials_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 15*r + 10 = 0 ∧ 
    r^3 + b*r^2 + 18*r + 12 = 0 ∧
    s^3 + a*s^2 + 15*s + 10 = 0 ∧ 
    s^3 + b*s^2 + 18*s + 12 = 0) →
  a = 3 ∧ b = 4 := by
sorry

end cubic_polynomials_common_roots_l3196_319695


namespace solution_range_l3196_319672

theorem solution_range (b : ℝ) : 
  let f := fun x : ℝ => x^2 - b*x - 5
  (f (-2) = 5) → 
  (f (-1) = -1) → 
  (f 4 = -1) → 
  (f 5 = 5) → 
  ∀ x : ℝ, f x = 0 → ((-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5)) :=
by sorry

end solution_range_l3196_319672


namespace perpendicular_to_parallel_line_l3196_319686

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_parallel_line 
  (α β : Plane) (m n : Line) 
  (h1 : α ≠ β) 
  (h2 : m ≠ n) 
  (h3 : perpendicular m α) 
  (h4 : parallel n α) : 
  perpendicular_lines m n :=
sorry

end perpendicular_to_parallel_line_l3196_319686


namespace unique_set_satisfying_condition_l3196_319689

theorem unique_set_satisfying_condition :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (a * b * c % d = 1) ∧
    (a * b * d % c = 1) ∧
    (a * c * d % b = 1) ∧
    (b * c * d % a = 1) →
    ({a, b, c, d} : Set ℕ) = {1, 2, 3, 4} :=
by sorry

end unique_set_satisfying_condition_l3196_319689


namespace sequence_comparison_l3196_319608

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ+ → ℝ :=
  fun n => a₁ + (n.val - 1) * d

def geometric_sequence (b₁ : ℝ) (q : ℝ) : ℕ+ → ℝ :=
  fun n => b₁ * q^(n.val - 1)

theorem sequence_comparison (a : ℕ+ → ℝ) (b : ℕ+ → ℝ) :
  (a 1 = 2) →
  (b 1 = 2) →
  (a 2 = 4) →
  (b 2 = 4) →
  (∀ n : ℕ+, a n = 2 * n.val) →
  (∀ n : ℕ+, b n = 2^n.val) →
  (∀ n : ℕ+, n ≥ 3 → a n < b n) :=
by sorry

end sequence_comparison_l3196_319608


namespace extra_fee_is_fifteen_l3196_319635

/-- Represents the data plan charges and fees -/
structure DataPlan where
  normalMonthlyCharge : ℝ
  promotionalRate : ℝ
  totalPaid : ℝ
  extraFee : ℝ

/-- Calculates the extra fee for going over the data limit -/
def calculateExtraFee (plan : DataPlan) : Prop :=
  let firstMonthCharge := plan.normalMonthlyCharge * plan.promotionalRate
  let regularMonthsCharge := plan.normalMonthlyCharge * 5
  let totalWithoutExtra := firstMonthCharge + regularMonthsCharge
  plan.extraFee = plan.totalPaid - totalWithoutExtra

/-- Theorem stating the extra fee is $15 given the problem conditions -/
theorem extra_fee_is_fifteen :
  ∃ (plan : DataPlan),
    plan.normalMonthlyCharge = 30 ∧
    plan.promotionalRate = 1/3 ∧
    plan.totalPaid = 175 ∧
    calculateExtraFee plan ∧
    plan.extraFee = 15 := by
  sorry

end extra_fee_is_fifteen_l3196_319635


namespace sin_1440_degrees_l3196_319699

theorem sin_1440_degrees : Real.sin (1440 * π / 180) = 0 := by
  sorry

end sin_1440_degrees_l3196_319699


namespace g_composition_of_three_l3196_319605

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 2

-- State the theorem
theorem g_composition_of_three : g (g (g 3)) = 107 := by
  sorry

end g_composition_of_three_l3196_319605


namespace system_solution_l3196_319678

theorem system_solution :
  let x : ℚ := -49/23
  let y : ℚ := 136/69
  (7 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 34) := by
  sorry

end system_solution_l3196_319678


namespace original_class_strength_l3196_319697

/-- Given an adult class, prove that the original strength was 12 students. -/
theorem original_class_strength
  (original_avg : ℝ)
  (new_students : ℕ)
  (new_avg : ℝ)
  (avg_decrease : ℝ)
  (h1 : original_avg = 40)
  (h2 : new_students = 12)
  (h3 : new_avg = 32)
  (h4 : avg_decrease = 4)
  : ∃ (x : ℕ), x = 12 ∧ 
    (x : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    ((x : ℝ) + new_students) * (original_avg - avg_decrease) :=
by
  sorry


end original_class_strength_l3196_319697


namespace science_club_committee_formation_l3196_319680

theorem science_club_committee_formation (total_members : ℕ) 
                                         (new_members : ℕ) 
                                         (committee_size : ℕ) 
                                         (h1 : total_members = 20) 
                                         (h2 : new_members = 10) 
                                         (h3 : committee_size = 4) :
  (Nat.choose total_members committee_size) - 
  (Nat.choose new_members committee_size) = 4635 :=
sorry

end science_club_committee_formation_l3196_319680


namespace bronze_status_donation_bound_l3196_319615

/-- Represents the fundraising status of the school --/
structure FundraisingStatus where
  goal : ℕ
  remaining : ℕ
  bronzeFamilies : ℕ
  silverFamilies : ℕ
  goldFamilies : ℕ

/-- Represents the donation tiers --/
structure DonationTiers where
  bronze : ℕ
  silver : ℕ
  gold : ℕ

/-- The Bronze Status donation is less than or equal to the remaining amount needed --/
theorem bronze_status_donation_bound (status : FundraisingStatus) (tiers : DonationTiers) :
  status.goal = 750 ∧
  status.remaining = 50 ∧
  status.bronzeFamilies = 10 ∧
  status.silverFamilies = 7 ∧
  status.goldFamilies = 1 ∧
  tiers.bronze ≤ tiers.silver ∧
  tiers.silver ≤ tiers.gold →
  tiers.bronze ≤ status.remaining :=
by sorry

end bronze_status_donation_bound_l3196_319615


namespace line_circle_intersection_range_l3196_319621

/-- The range of b for which the line y = x + b intersects the circle (x-2)^2 + (y-3)^2 = 4
    within the constraints 0 ≤ x ≤ 4 and 1 ≤ y ≤ 3 -/
theorem line_circle_intersection_range :
  ∀ b : ℝ,
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧
   y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔
  (1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3) :=
by sorry

end line_circle_intersection_range_l3196_319621


namespace scallops_per_pound_is_eight_l3196_319690

/-- The number of jumbo scallops that weigh one pound -/
def scallops_per_pound : ℕ := by sorry

/-- The cost of one pound of jumbo scallops in dollars -/
def cost_per_pound : ℕ := 24

/-- The number of scallops paired per person -/
def scallops_per_person : ℕ := 2

/-- The number of people Nate is cooking for -/
def number_of_people : ℕ := 8

/-- The total cost of scallops for Nate in dollars -/
def total_cost : ℕ := 48

theorem scallops_per_pound_is_eight :
  scallops_per_pound = 8 := by sorry

end scallops_per_pound_is_eight_l3196_319690


namespace marbles_after_2000_steps_l3196_319614

/-- Represents the state of baskets with marbles -/
def BasketState := List Nat

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the marble placement process for a given number of steps -/
def simulateMarblePlacement (steps : Nat) : BasketState :=
  sorry

/-- Counts the total number of marbles in a given basket state -/
def countMarbles (state : BasketState) : Nat :=
  sorry

/-- Theorem stating that the number of marbles after 2000 steps
    is equal to the sum of digits in the base-6 representation of 2000 -/
theorem marbles_after_2000_steps :
  countMarbles (simulateMarblePlacement 2000) = sumDigits (toBase6 2000) :=
by sorry

end marbles_after_2000_steps_l3196_319614


namespace complex_equality_l3196_319612

theorem complex_equality (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b + Complex.I →
  a + b = 3 := by
sorry

end complex_equality_l3196_319612


namespace car_speed_comparison_l3196_319668

/-- Given two cars A and B that travel the same distance, where:
    - Car A travels 1/3 of the distance at u mph, 1/3 at v mph, and 1/3 at w mph
    - Car B travels 1/3 of the time at u mph, 1/3 at v mph, and 1/3 at w mph
    - Average speed of Car A is x mph
    - Average speed of Car B is y mph
    This theorem proves that the average speed of Car A is less than or equal to the average speed of Car B. -/
theorem car_speed_comparison 
  (u v w : ℝ) 
  (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (x y : ℝ) 
  (hx : x = 3 / (1/u + 1/v + 1/w)) 
  (hy : y = (u + v + w) / 3) : 
  x ≤ y := by
sorry

end car_speed_comparison_l3196_319668


namespace remainder_55_pow_55_plus_15_mod_8_l3196_319619

theorem remainder_55_pow_55_plus_15_mod_8 : (55^55 + 15) % 8 = 6 := by
  sorry

end remainder_55_pow_55_plus_15_mod_8_l3196_319619


namespace max_students_is_18_l3196_319604

/-- Represents the structure of Ms. Gregory's class -/
structure ClassStructure where
  boys : ℕ
  girls : ℕ
  science_club : ℕ
  math_club : ℕ

/-- Checks if the given class structure satisfies all conditions -/
def is_valid_structure (c : ClassStructure) : Prop :=
  3 * c.boys = 4 * c.science_club ∧ 
  2 * c.girls = 3 * c.science_club ∧ 
  c.math_club = 2 * c.science_club ∧
  c.boys + c.girls = c.science_club + c.math_club

/-- The maximum number of students in Ms. Gregory's class -/
def max_students : ℕ := 18

/-- Theorem stating that the maximum number of students is 18 -/
theorem max_students_is_18 : 
  ∀ c : ClassStructure, is_valid_structure c → c.boys + c.girls ≤ max_students :=
by
  sorry

#check max_students_is_18

end max_students_is_18_l3196_319604


namespace total_chewing_gums_l3196_319610

theorem total_chewing_gums (mary sam sue : ℕ) : 
  mary = 5 → sam = 10 → sue = 15 → mary + sam + sue = 30 := by
  sorry

end total_chewing_gums_l3196_319610


namespace roots_polynomial_d_values_l3196_319677

theorem roots_polynomial_d_values (u v c d : ℝ) : 
  (∃ w : ℝ, {u, v, w} = {x | x^3 + c*x + d = 0}) ∧
  (∃ w : ℝ, {u+3, v-2, w} = {x | x^3 + c*x + (d+120) = 0}) →
  d = 84 ∨ d = -25 := by
sorry

end roots_polynomial_d_values_l3196_319677


namespace other_colors_correct_l3196_319600

/-- Represents a school with its student data -/
structure School where
  total_students : ℕ
  blue_percent : ℚ
  red_percent : ℚ
  green_percent : ℚ
  blue_red_percent : ℚ
  blue_green_percent : ℚ
  red_green_percent : ℚ

/-- Calculates the number of students wearing other colors -/
def other_colors (s : School) : ℕ :=
  s.total_students - (s.total_students * (s.blue_percent + s.red_percent + s.green_percent - 
    s.blue_red_percent - s.blue_green_percent - s.red_green_percent)).ceil.toNat

/-- The first school's data -/
def school1 : School := {
  total_students := 800,
  blue_percent := 30/100,
  red_percent := 20/100,
  green_percent := 10/100,
  blue_red_percent := 5/100,
  blue_green_percent := 3/100,
  red_green_percent := 2/100
}

/-- The second school's data -/
def school2 : School := {
  total_students := 700,
  blue_percent := 25/100,
  red_percent := 25/100,
  green_percent := 20/100,
  blue_red_percent := 10/100,
  blue_green_percent := 5/100,
  red_green_percent := 3/100
}

/-- The third school's data -/
def school3 : School := {
  total_students := 500,
  blue_percent := 1/100,
  red_percent := 1/100,
  green_percent := 1/100,
  blue_red_percent := 1/2/100,
  blue_green_percent := 1/2/100,
  red_green_percent := 1/2/100
}

/-- Theorem stating the correct number of students wearing other colors in each school -/
theorem other_colors_correct :
  other_colors school1 = 400 ∧
  other_colors school2 = 336 ∧
  other_colors school3 = 475 := by
  sorry

end other_colors_correct_l3196_319600


namespace sequence_inequality_range_l3196_319643

/-- Given a sequence a_n with sum S_n, prove the range of t -/
theorem sequence_inequality_range (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) : 
  (∀ n : ℕ, 2 * S n = (n + 1) * a n) →  -- Condition: 2S_n = (n+1)a_n
  (a 1 = 1) →  -- Condition: a_1 = 1
  (∀ n : ℕ, n ≥ 2 → a n = n) →  -- Derived from conditions
  (t > 0) →  -- Condition: t > 0
  (∃! n : ℕ, n > 0 ∧ a n^2 - t * a n - 2 * t^2 < 0) →  -- Condition: unique positive n satisfying inequality
  t ∈ Set.Ioo (1/2 : ℝ) 1 :=  -- Conclusion: t is in the open interval (1/2, 1]
sorry

end sequence_inequality_range_l3196_319643


namespace intersection_of_A_and_B_l3196_319606

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3^x ∧ x ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 1 := by sorry

end intersection_of_A_and_B_l3196_319606


namespace reading_time_difference_l3196_319667

def xanthia_speed : ℝ := 120
def molly_speed : ℝ := 60
def book_pages : ℝ := 300

theorem reading_time_difference : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
  sorry

end reading_time_difference_l3196_319667


namespace pictures_from_phone_l3196_319645

-- Define the problem parameters
def num_albums : ℕ := 3
def pics_per_album : ℕ := 2
def camera_pics : ℕ := 4

-- Define the total number of pictures
def total_pics : ℕ := num_albums * pics_per_album

-- Define the number of pictures from the phone
def phone_pics : ℕ := total_pics - camera_pics

-- Theorem statement
theorem pictures_from_phone : phone_pics = 2 := by
  sorry

end pictures_from_phone_l3196_319645


namespace total_wheels_is_150_l3196_319602

/-- The total number of wheels Naomi saw at the park -/
def total_wheels : ℕ :=
  let regular_bikes := 7
  let children_bikes := 11
  let tandem_bikes_4 := 5
  let tandem_bikes_6 := 3
  let unicycles := 4
  let tricycles := 6
  let training_wheel_bikes := 8

  let regular_bike_wheels := 2
  let children_bike_wheels := 4
  let tandem_bike_4_wheels := 4
  let tandem_bike_6_wheels := 6
  let unicycle_wheels := 1
  let tricycle_wheels := 3
  let training_wheel_bike_wheels := 4

  regular_bikes * regular_bike_wheels +
  children_bikes * children_bike_wheels +
  tandem_bikes_4 * tandem_bike_4_wheels +
  tandem_bikes_6 * tandem_bike_6_wheels +
  unicycles * unicycle_wheels +
  tricycles * tricycle_wheels +
  training_wheel_bikes * training_wheel_bike_wheels

theorem total_wheels_is_150 : total_wheels = 150 := by
  sorry

end total_wheels_is_150_l3196_319602


namespace no_simultaneous_squares_l3196_319659

theorem no_simultaneous_squares : ¬ ∃ (x y : ℕ), 
  ∃ (a b : ℕ), (x^2 + 2*y = a^2) ∧ (y^2 + 2*x = b^2) := by
  sorry

end no_simultaneous_squares_l3196_319659


namespace jimmy_cards_l3196_319632

/-- 
Given:
- Jimmy gives 3 cards to Bob
- Jimmy gives twice as many cards to Mary as he gave to Bob
- Jimmy has 9 cards left after giving away cards

Prove that Jimmy initially had 18 cards.
-/
theorem jimmy_cards : 
  ∀ (cards_to_bob cards_to_mary cards_left initial_cards : ℕ),
  cards_to_bob = 3 →
  cards_to_mary = 2 * cards_to_bob →
  cards_left = 9 →
  initial_cards = cards_to_bob + cards_to_mary + cards_left →
  initial_cards = 18 := by
sorry


end jimmy_cards_l3196_319632


namespace amount_subtracted_l3196_319611

theorem amount_subtracted (number : ℝ) (subtracted_amount : ℝ) : 
  number = 70 →
  (number / 2) - subtracted_amount = 25 →
  subtracted_amount = 10 := by
sorry

end amount_subtracted_l3196_319611


namespace no_valid_ratio_l3196_319638

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  b : ℝ  -- Length of the larger base
  a : ℝ  -- Length of the smaller base
  h : ℝ  -- Height of the trapezoid
  is_positive : 0 < b
  smaller_base_eq_diagonal : a = h
  altitude_eq_larger_base : h = b

/-- Theorem stating that no valid ratio exists between the bases of the described trapezoid -/
theorem no_valid_ratio (t : IsoscelesTrapezoid) : False :=
sorry

end no_valid_ratio_l3196_319638


namespace chocolate_bars_count_l3196_319675

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 20

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := num_small_boxes * bars_per_small_box

theorem chocolate_bars_count : total_bars = 500 := by
  sorry

end chocolate_bars_count_l3196_319675


namespace goods_train_length_l3196_319642

/-- Calculates the length of a goods train given the speeds of two trains
    traveling in opposite directions and the time taken for the goods train
    to pass a stationary observer in the other train. -/
theorem goods_train_length
  (speed_train : ℝ)
  (speed_goods : ℝ)
  (pass_time : ℝ)
  (h1 : speed_train = 15)
  (h2 : speed_goods = 97)
  (h3 : pass_time = 9)
  : ∃ (length : ℝ), abs (length - 279.99) < 0.01 :=
by
  sorry

end goods_train_length_l3196_319642


namespace nonagon_non_adjacent_segments_l3196_319617

theorem nonagon_non_adjacent_segments (n : ℕ) (h : n = 9) : 
  (n * (n - 1)) / 2 - n = 27 := by
  sorry

end nonagon_non_adjacent_segments_l3196_319617
